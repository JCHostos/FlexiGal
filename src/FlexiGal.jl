module FlexiGal
using StaticArrays
using LinearAlgebra
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
include("WeakForm_Macro.jl")
export create_model, BackgroundIntegration, Flexi_Space, Influence_Domains, FlexiFunction, ApproxSpace, Solve, build_space,
    Domain_Measure, Get_Point_Values, ∇, Internal_Product, Integrate, ∫, ⋅, ⊙, PointSet, δ, Bilinear_Assembler, Linear_Assembler, VectorField, Linear_Problem, plot_field, Assemble_Vector,
    Triangulation, IntegrationSet, Get_space_from_IntegrationSet, tr, Id, NL_Solver, @WeakForm, @NL_WeakForm, NonLinearOperator, Get_Nodal_Values, Get_Measures, build_space,
    Reassemble_Vector!, set_tag!, MultiApproxSpace, evaluate_at_point,  extract_FlexiFunction, set_boundary_tag!, EFGInterpolatedField, Get_Interpolated_Values
struct Triangulation
    model::model
    tag::String
end
struct IntegrationSet
    tri::Triangulation
    degree::Int
end

struct DomainMeasure
    tag::String
    gs::Matrix{Float64}
    degree::Int
end

struct PointSet
    coords::Matrix{Float64}   # n_points × dim
    degree::Int
end

PointSet(coords::Matrix{Float64}) = PointSet(coords, 0)

PointSet(point::AbstractVector{<:Real}) = PointSet(reshape(collect(Float64.(point)), 1, :))

function PointSet(points::AbstractVector{<:AbstractVector{<:Real}})
    n = length(points); dim = length(points[1])
    M = Matrix{Float64}(undef, n, dim)
    for (i, p) in enumerate(points); M[i, :] .= p; end
    return PointSet(M)
end


function BackgroundIntegration(iset::IntegrationSet, method::Symbol; order=1)
    model = iset.tri.model
    tag = iset.tri.tag
    degree = iset.degree
    dim = model.dim
    conn, cat = get_entity_info(model, tag)
    numvxcell = size(conn, 2)
    domain_cat = (dim == 2) ? :faces : :volumes
    is_boundary = (cat !== domain_cat)
    if !is_boundary
        if dim == 2
            if numvxcell == 3
                gauss = ptriangle(degree)
            elseif numvxcell == 4
                gauss = pgauss(degree)
            else
                error("Elemento de dominio 2D no soportado con $numvxcell nodos")
            end
        elseif dim == 3
            if numvxcell == 4
                gauss = ptetra(degree)
            elseif numvxcell == 8
                gauss = pgauss(degree)
            else
                error("Elemento de dominio 3D no soportado con $numvxcell nodos")
            end
        end
    else
        if dim == 2
            @assert numvxcell == 2 "Las celdas de frontera en 2D deben tener 2 nodos"
            gauss = pgauss(degree) 
        elseif dim == 3
            if numvxcell == 3
                gauss = ptriangle(degree)
            elseif numvxcell == 4
                gauss = pgauss(degree)
            else
                error("Celdas de frontera en 3D no soportadas con $numvxcell nodos")
            end
        end
    end
    if method === :FEM
        if !is_boundary
            gs, PHI, DPHI, DOM = egauss_fem(model.x, conn, gauss; order=order)
        else
            gs, PHI, DPHI, DOM = egauss_bound_fem(model.x, conn, gauss; order=order)
        end
        return DomainMeasure(tag, gs, degree), PHI, DPHI, DOM
    elseif method === :EFG
        if !is_boundary
            gs = egauss(model.x, conn, gauss)
        else
            gs = egauss_bound(model.x, conn, gauss)
        end
        return DomainMeasure(tag, gs, degree)     
    else
        error("Método desconocido: $method")
    end
end

@inline merge(measure::DomainMeasure; tag="merged") = measure

@inline function merge(measures::Vector{DomainMeasure}; tag::String="merged")
    isempty(measures) && error("No DomainMeasure to merge")
    all_gs = vcat([m.gs for m in measures]...)
    return DomainMeasure(tag, all_gs, measures[1].degree)
end

struct ApproxSpace{DV}
    model::model
    triangulations::Vector{Triangulation}
    Field_Type::Type
    dmax::Union{Float64,Vector{Float64}}
    shape::Symbol
    technique::Symbol
    methods::Dict{String,Symbol}
    dm_map::Union{Function,Nothing}
    gamma::Float64
    Cq::Float64
    Dirichlet_Entities::Vector{Triangulation}
    Dirichlet_Values::Vector{DV}
    Dirichlet_Masks::Vector{Vector{Bool}}
    constraint::Symbol
    order::Int
    cell_based::Bool
end

# El constructor que usaremos en tu script
function ApproxSpace(model, triangs, Field_Type; dmax=1.5, shape::Symbol=:rectangular, technique=:IMLS, gamma=5.0, Cq=4.3,
    method=:EFG, dm_map=nothing,
    Dirichlet_Entities=Triangulation[],
    Dirichlet_Values=Float64[],Dirichlet_Masks=Vector{Vector{Bool}}(),constraint=:none, order=1, cell_based = false)
    T = triangs isa Triangulation ? [triangs] : triangs
    DT = Dirichlet_Entities isa Triangulation ? [Dirichlet_Entities] : Dirichlet_Entities
    DV = eltype(Dirichlet_Values)
    D = field_dim(Field_Type)
    final_masks = Vector{Vector{Bool}}()
    if isempty(Dirichlet_Masks)
        final_masks = [fill(true, D) for _ in 1:length(DT)]
    else
        if Dirichlet_Masks isa Vector{Bool}
            final_masks = [convert(Vector{Bool}, Dirichlet_Masks)]
        else
            final_masks = [convert(Vector{Bool}, m) for m in Dirichlet_Masks]
        end
    end
    @assert length(final_masks) == length(DT) "La cantidad de máscaras ($(length(final_masks))) debe coincidir con la cantidad de Dirichlet_Entities ($(length(DT)))"

    methods_dict = Dict{String,Symbol}()
    if method isa Symbol
        for t in T
            methods_dict[t.tag] = method
        end
    elseif method isa Vector{Symbol}
        @assert length(T) == length(method) "La longitud del vector 'method' debe coincidir con la de triangulations"
        for (i, t) in enumerate(T)
            methods_dict[t.tag] = method[i]
        end
    else
        error("method debe ser un Symbol (ej: :FEM) o un Vector{Symbol} (ej: [:EFG, :FEM, :FEM])")
    end
    default_m = method isa Symbol ? method : method[1]
    for dt in DT
        if !haskey(methods_dict, dt.tag)
            methods_dict[dt.tag] = default_m
        end
    end
    return ApproxSpace{DV}(model, T, Field_Type, dmax, shape, technique, methods_dict,dm_map, gamma,Cq, DT, Dirichlet_Values, final_masks, constraint, order, cell_based)
end

struct FlexiSpace{DG}
    domain::Dict{String,Tuple{
        Vector{Vector{Float64}},
        Vector{Vector{SVector{DG,Float64}}},
        Vector{Vector{Int}}
    }}
    boundary::Dict{String,Tuple{
        Vector{Vector{Float64}},
        Vector{Vector{SVector{DG,Float64}}},
        Vector{Vector{Int}}
    }}
    Field_Type::Type
    Measures::Vector{DomainMeasure}
    nnodes::Int
end

function Influence_Domains_CellBased(model, dmax::Real; shape::Symbol=:rectangular, dm_map=nothing)
    x = model.x
    conn = model.conn
    nnodes = size(x, 1)
    ncells = size(conn, 1)
    nvxcell = size(conn, 2)
    dim = size(x, 2)

    nodal_measure = zeros(Float64, nnodes)
    nodal_count = zeros(Int, nnodes)

    # --- Precálculo de derivadas locales para celdas de 4 y 8 nodos ---
    if nvxcell == 4 && dim == 2
        # Cuadriláteros 2D: la evaluación en (0,0) es EXACTA para el área de cualquier cuadrilátero distorsionado
        dNloc_quad = 0.25 * [-1.0  1.0  1.0 -1.0;
                             -1.0 -1.0  1.0  1.0]

    elseif nvxcell == 8 && dim == 3
        # Hexaedros 3D: Integración de Gauss 2x2x2 para volumen EXACTO en celdas distorsionadas
        gp_val = 1.0 / sqrt(3.0)
        gp_coords = (-gp_val, gp_val)

        # Coordenadas locales de los 8 nodos de un hexaedro canónico [-1, 1]^3
        xi_n   = (-1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0)
        eta_n  = (-1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0)
        zeta_n = (-1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0)

        dNloc_hex = zeros(Float64, 3, 8, 8) # (dirección, nodo, punto_gauss)
        gp_idx = 1
        for gz in gp_coords, gy in gp_coords, gx in gp_coords
            for k in 1:8
                dNloc_hex[1, k, gp_idx] = 0.125 * xi_n[k]   * (1.0 + eta_n[k]*gy)  * (1.0 + zeta_n[k]*gz)
                dNloc_hex[2, k, gp_idx] = 0.125 * eta_n[k]  * (1.0 + xi_n[k]*gx)   * (1.0 + zeta_n[k]*gz)
                dNloc_hex[3, k, gp_idx] = 0.125 * zeta_n[k] * (1.0 + xi_n[k]*gx)   * (1.0 + eta_n[k]*gy)
            end
            gp_idx += 1
        end
    end

    @inbounds for e in 1:ncells
        c_meas = 0.0
        scale_factor = 1.0

        if nvxcell == 3 && dim == 2
            # 1) TRIÁNGULOS (2D)
            n1, n2, n3 = conn[e, 1], conn[e, 2], conn[e, 3]
            x1, y1 = x[n1, 1], x[n1, 2]
            x2, y2 = x[n2, 1], x[n2, 2]
            x3, y3 = x[n3, 1], x[n3, 2]

            # Área exacta del triángulo
            c_meas = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            # Triángulo equilátero: A = (√3/4)*a² => a² = (4/√3)*A
            scale_factor = 4.0 / sqrt(3.0)

        elseif nvxcell == 4 && dim == 2
            # 2) CUADRILÁTEROS (2D)
            J11, J12, J21, J22 = 0.0, 0.0, 0.0, 0.0
            for k in 1:4
                nid = conn[e, k]
                xk, yk = x[nid, 1], x[nid, 2]
                J11 += dNloc_quad[1, k] * xk
                J12 += dNloc_quad[1, k] * yk
                J21 += dNloc_quad[2, k] * xk
                J22 += dNloc_quad[2, k] * yk
            end
            c_meas = 4.0 * abs(J11 * J22 - J12 * J21)
            scale_factor = 1.0

        elseif nvxcell == 4 && dim == 3
            # 3) TETRAEDROS (3D)
            n1, n2, n3, n4 = conn[e, 1], conn[e, 2], conn[e, 3], conn[e, 4]
            x1, y1, z1 = x[n1, 1], x[n1, 2], x[n1, 3]
            x2, y2, z2 = x[n2, 1], x[n2, 2], x[n2, 3]
            x3, y3, z3 = x[n3, 1], x[n3, 2], x[n3, 3]
            x4, y4, z4 = x[n4, 1], x[n4, 2], x[n4, 3]

            x21, y21, z21 = x2 - x1, y2 - y1, z2 - z1
            x31, y31, z31 = x3 - x1, y3 - y1, z3 - z1
            x41, y41, z41 = x4 - x1, y4 - y1, z4 - z1

            # Volumen exacto del tetraedro
            c_meas = (1.0 / 6.0) * abs(
                x21 * (y31 * z41 - z31 * y41) -
                y21 * (x31 * z41 - z31 * x41) +
                z21 * (x31 * y41 - y31 * x41)
            )
            # Tetraedro regular: V = (√2/12)*a³ => a³ = 6*√2*V ≈ 8.48528*V
            scale_factor = 6.0 * sqrt(2.0)

        elseif nvxcell == 8 && dim == 3
            # 4) HEXAEDROS (3D) - Gauss 2x2x2 para celdas distorsionadas
            c_meas = 0.0
            for gp in 1:8
                J11, J12, J13 = 0.0, 0.0, 0.0
                J21, J22, J23 = 0.0, 0.0, 0.0
                J31, J32, J33 = 0.0, 0.0, 0.0
                for k in 1:8
                    nid = conn[e, k]
                    xk, yk, zk = x[nid, 1], x[nid, 2], x[nid, 3]
                    dNx = dNloc_hex[1, k, gp]
                    dNy = dNloc_hex[2, k, gp]
                    dNz = dNloc_hex[3, k, gp]
                    J11 += dNx * xk; J12 += dNx * yk; J13 += dNx * zk
                    J21 += dNy * xk; J22 += dNy * yk; J23 += dNy * zk
                    J31 += dNz * xk; J32 += dNz * yk; J33 += dNz * zk
                end
                detJ = J11 * (J22 * J33 - J23 * J32) - J12 * (J21 * J33 - J23 * J31) + J13 * (J21 * J32 - J22 * J31)
                c_meas += abs(detJ)
            end
            scale_factor = 1.0
        end

        for k in 1:nvxcell
            nid = conn[e, k]
            nodal_measure[nid] += (c_meas * scale_factor)
            nodal_count[nid] += 1
        end
    end

    if shape === :rectangular
        dm = zeros(Float64, nnodes, dim)
        @inbounds for i in 1:nnodes
            avg_meas = nodal_measure[i] / nodal_count[i]
            h_local = (dim == 2) ? sqrt(avg_meas) : cbrt(avg_meas)
            dm[i, :] .= dmax * h_local
        end

        if dm_map !== nothing
            @inbounds for i in 1:nnodes
                scale = dm_map(@view model.x0[i, :])
                if isa(scale, AbstractVector) || isa(scale, Tuple)
                    for d in 1:dim
                        dm[i, d] *= scale[d]
                    end
                else
                    dm[i, :] .*= scale
                end
            end
        end
        return dm

    elseif shape === :circular || shape === :spherical
        dm_vec = zeros(Float64, nnodes)
        @inbounds for i in 1:nnodes
            avg_meas = nodal_measure[i] / nodal_count[i]
            h_local = (dim == 2) ? sqrt(avg_meas) : cbrt(avg_meas)
            dm_vec[i] = dmax * h_local
        end
        return dm_vec
    else
        error("Forma de dominio de influencia no soportada: $shape. Formas soportadas: :rectangular, :circular, :spherical.")
    end
end

function Influence_Domains(model::model{DG}, Domain::Tuple, Divisions::Tuple, dmax::Union{Real,AbstractVector{<:Real}}; shape::Symbol=:rectangular, dm_map::Union{Function,Nothing}=nothing) where DG
    dim = size(model.x, 2)
    dvec = isa(dmax, Real) ? fill(dmax, dim) : dmax
    nnodes = size(model.x, 1)
    if shape === :rectangular
        dm = zeros(nnodes, dim)
        for d in 1:dim
            dm[:, d] .= dvec[d] * Domain[d] / Divisions[d]
        end
        if dm_map !== nothing
            @inbounds for i in 1:nnodes
                # El factor se calcula según la posición original x0
                scale = dm_map(@view model.x0[i, :])

                if isa(scale, AbstractVector) || isa(scale, Tuple)
                    for d in 1:dim
                        dm[i, d] *= scale[d]
                    end
                else
                    # Si devuelve un escalar, escala todas las dimensiones igual
                    dm[i, :] .*= scale
                end
            end
        end
        return dm
    elseif shape === :circular || shape === :spherical
        spacing_avg = sum(Domain ./ Divisions) / dim
        radius = dvec[1] * spacing_avg
        if dm_map !== nothing
            @warn "Transformation not applied because the influence domain is circular. Use shape=:rectangular for transformed domains."
        end
        return fill(radius, nnodes)
    else
        error("Unknown shape of influence domain: $shape. Shapes supported are: rectangular, circular, spherical, cylindrical.")
    end
end

struct FlexiMeasure{DG}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Vector{SVector{DG,Float64}}} # Formato optimizado
    DOM::Vector{Vector{Int}}
    nnodes::Int
end
function Flexi_Measure(measures::Union{DomainMeasure,AbstractVector{<:DomainMeasure}}, Shape_Functions::FlexiSpace)
    measures_list = isa(measures, DomainMeasure) ? [measures] : measures
    DG = typeof(Shape_Functions).parameters[1]
    all_PHI = Vector{Vector{Float64}}()
    all_DPHI = Vector{Vector{SVector{DG,Float64}}}()
    all_DOM = Vector{Vector{Int}}()
    for sm in measures_list
        tag = sm.tag
        shape = haskey(Shape_Functions.domain, tag) ? Shape_Functions.domain[tag] :
                haskey(Shape_Functions.boundary, tag) ? Shape_Functions.boundary[tag] :
                error("No shape functions for tag '$tag'")
        PHI, DPHI, DOM = shape
        append!(all_PHI, PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM, DOM)
    end
    return FlexiMeasure{DG}(all_PHI, all_DPHI, all_DOM, Shape_Functions.nnodes)
end

function get_topology_layout(conn::Matrix{Int}, dim::Int)
    nvxcell = size(conn, 2)
    # Triángulos (3) y Tetraedros (4 en 3D) -> :linear
    # Cuadriláteros (4 en 2D) y Hexaedros (8 en 3D) -> :multilinear
    return (nvxcell == 3 || (nvxcell == 4 && dim == 3)) ? :linear : :multilinear
end

function build_space(recipe::ApproxSpace, isets::Vector{IntegrationSet})
    m = recipe.model::model
    dim = size(m.x, 2)
    DG = dim
    Layout = get_topology_layout(m.conn,dim)
    T_Shape = Tuple{Vector{Vector{Float64}},Vector{Vector{SVector{DG,Float64}}},Vector{Vector{Int}}}
    results_domain = Dict{String,T_Shape}()
    results_boundary = Dict{String,T_Shape}()
    gs_list = DomainMeasure[]
    shape = recipe.shape
    technique = recipe.technique
    dm_map = recipe.dm_map
    gamma = recipe.gamma
    Cq=recipe.Cq
    needs_efg = any(get(recipe.methods, iset.tri.tag, :EFG) === :EFG for iset in isets)
    if needs_efg
        if recipe.cell_based
            dm = Influence_Domains_CellBased(m, recipe.dmax; shape=shape, dm_map=dm_map)
        else
            dm = Influence_Domains(m, m.domain, m.divisions, recipe.dmax; shape=shape, dm_map=dm_map)
        end
    else
        dm = Matrix{Float64}(undef, 0, 0)
    end
    for iset in isets
        tag = iset.tri.tag
        method = get(recipe.methods, tag, :EFG)
        conn, cat = get_entity_info(m, tag)
        domain_cat = (dim == 2) ? :faces : :volumes
        gs_type = (cat === domain_cat) ? :domain : :boundary
        if method === :FEM
            meas, PHI, DPHI, DOM = BackgroundIntegration(iset, :FEM; order=recipe.order)
        else
            meas = BackgroundIntegration(iset, :EFG)
            if gs_type === :domain
                global_node_ids = sort!(unique(view(conn, :)))
                PHI, DPHI, DOM_local = SHAPE_FUN(meas.gs, m.x[global_node_ids,:], dm[global_node_ids,:]; shape=shape, technique=technique, Layout=Layout, gamma=gamma, Cq=Cq)
                DOM = [global_node_ids[idx] for idx in DOM_local]
            else
                PHI, DPHI, DOM = SHAPE_FUN(meas.gs, m.x, dm; shape=shape, technique=technique, Layout=Layout, gamma=gamma,Cq=Cq)
            end
        end
        push!(gs_list, meas)
        target_dict = (gs_type === :domain) ? results_domain : results_boundary
        target_dict[tag] = (PHI, DPHI, DOM)
    end
    return FlexiSpace{DG}(results_domain, results_boundary, recipe.Field_Type, gs_list, m.nnodes)
end

function build_point_space(recipe::ApproxSpace, pset::PointSet)
    m = recipe.model
    dim = size(m.x, 2)
    DG = dim
    Layout = get_topology_layout(m.conn, dim)
    dm = recipe.cell_based ?
        Influence_Domains_CellBased(m, recipe.dmax; shape=recipe.shape, dm_map=recipe.dm_map) :
        Influence_Domains(m, m.domain, m.divisions, recipe.dmax; shape=recipe.shape, dm_map=recipe.dm_map)

    PHI, DPHI, DOM = SHAPE_FUN(pset.coords, m.x, dm; shape=recipe.shape, technique=recipe.technique, Layout=Layout, gamma=recipe.gamma)

    n_pts = size(pset.coords, 1)
    gs = hcat(pset.coords, ones(n_pts), ones(n_pts))   # peso=1, jacobiano=1 -> "sin cuadratura"
    tag = "__point_load__"
    measure = DomainMeasure(tag, gs, 0)
    T_Shape = Tuple{Vector{Vector{Float64}}, Vector{Vector{SVector{DG, Float64}}}, Vector{Vector{Int}}}
    domain_dict = Dict{String, T_Shape}(tag => (PHI, DPHI, DOM))
    boundary_dict = Dict{String, T_Shape}()
    return FlexiSpace(domain_dict, boundary_dict, recipe.Field_Type, [measure], m.nnodes)
end

struct MultiApproxSpace
    spaces::Vector{ApproxSpace}
    offsets::Vector{Int}
    ndofs::Int
    function MultiApproxSpace(spaces::Vector{<:ApproxSpace}) # Usamos <: para permitir diferentes DV
        offsets = Int[]
        current_offset = 0
        for i in 1:length(spaces)
            push!(offsets, current_offset)
            d = field_dim(spaces[i].Field_Type)
            if spaces[i].order == 0
            n_elements = spaces[i].model.ncells
            current_offset += n_elements * d
            else
            current_offset += spaces[i].model.nnodes * d
            end
        end
        return new(convert(Vector{ApproxSpace}, spaces), offsets, current_offset)
    end
end

MultiApproxSpace(args...) = MultiApproxSpace([args...])

function global_dofs(mspace::MultiApproxSpace, field_idx::Int, local_nodes::Vector{Int})
    space = mspace.spaces[field_idx]
    offset = mspace.offsets[field_idx]
    d = field_dim(space.Field_Type)
    if d == 1
        return local_nodes .+ offset
    else
        total_dofs = length(local_nodes) * d
        indices = Vector{Int}(undef, total_dofs)
        for (i, node) in enumerate(local_nodes)
            base = offset + (node - 1) * d
            for j in 1:d
                indices[(i-1)*d + j] = base + j
            end
        end
        return indices
    end
end

include("Fields_Operations.jl")
include("Assembling_Operators.jl")
include("Solvers.jl")
include("Plots.jl")
end