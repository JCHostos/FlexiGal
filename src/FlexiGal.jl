module FlexiGal
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
export create_model, BackgroundIntegration, EFGSpace, Influence_Domains, AssembleEFG, EFGFunction,
    Domain_Measure, Get_Point_Values, ∇, Internal_Product, Integrate, ∫, ⋅, *, EFG_Measure, Bilinear_Assembler, Merge_Measures
struct DomainMeasure
    tag::String
    gs::Matrix{Float64}
end
function BackgroundIntegration(model::EFGmodel, tag::String, degree::Int)
    conn = get_entity(model, tag)
    gauss = pgauss(degree)
    dim = size(model.x, 2)
    if size(conn, 2) == 2^dim
        gs = egauss(model.x, conn, gauss)
    elseif size(conn, 2) == 2^(dim - 1)
        gs = egauss_bound(model.x, conn, gauss)
    end
    conn= nothing
    return DomainMeasure(tag, gs)
end
function Merge_Measures(model::EFGmodel, measures::Vector{DomainMeasure}; tag::String)
    # 1. Concatenar gauss points
    gs_merged = vcat([m.gs for m in measures]...)

    # 2. Buscar conectividades originales de cada tag
    conns = Matrix{Int}[]
    for m in measures
        conn = get_entity(model, m.tag)
        if conn isa Int
            push!(conns, [conn])  # en caso raro de que sea un solo nodo
        else
            push!(conns, conn)
        end
    end
    conn_merged = vcat(conns...)

    # 3. Crear nuevo DomainMeasure
    new_measure = DomainMeasure(tag, gs_merged)

    # 4. Insertar en el model.entities en la categoría correcta
    new_entities = deepcopy(model.entities)

    # buscar en qué subdiccionario estaba el primer tag
    parent_key = nothing
    for (etype, subdict) in model.entities
        if haskey(subdict, measures[1].tag)
            parent_key = etype
            break
        end
    end
    parent_key === nothing && error("No se encontró ninguna categoría para el tag $(measures[1].tag)")

    new_entities[parent_key][tag] = conn_merged

    # 5. Nuevo model actualizado
    new_model = EFGmodel(model.x, model.conn, new_entities,
                         model.dim, model.ncells, model.nnodes)
     conn_merged=gs_merged=nothing
    return new_measure, new_model
end
# Constructing Shape Functions given model, Measures for Integration and Influence domains
struct EFGSpace
    domain::Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}
    boundary::Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}
    nnodes::Int
end

function EFGSpace(model::EFGmodel, gs_list::Union{DomainMeasure, Vector{DomainMeasure}}, dm::Matrix{Float64})
    # Forzar que gs_list siempre sea un vector
    gs_list = isa(gs_list, DomainMeasure) ? [gs_list] : gs_list

    x = model.x
    results_domain = Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}()
    results_boundary = Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}()
    nnodes, dim = size(x)

    for measure in gs_list
        tag = measure.tag
        gs = measure.gs
        conn = get_entity(model, tag)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary

        PHI, DPHI, DOM = SHAPE_FUN(gs, x, dm)

        if gs_type == :domain
            results_domain[tag] = (PHI, DPHI, DOM)
        else
            results_boundary[tag] = (PHI, DPHI, DOM)
        end
    end
    gs_list=nothing

    return EFGSpace(results_domain, results_boundary, nnodes)
end
# Defining Influence Domains (Ongoing Development)
function Influence_Domains(model::EFGmodel, Domain::Tuple, Divisions::Tuple, dmax::Float64)
    dim = size(model.x, 2)
    dm = zeros(size(model.x, 1), dim)
    if dim == 2
        Lx, Ly = Domain
        Nx, Ny = Divisions
        dm[:, 1] .= dmax * Lx / Nx
        dm[:, 2] .= dmax * Ly / Ny
    elseif dim == 3
        Lx, Ly, Lz = Domain
        Nx, Ny, Nz = Divisions
        dm[:, 1] .= dmax * Lx / Nx
        dm[:, 2] .= dmax * Ly / Ny
        dm[:, 3] .= dmax * Lz / Nz
    end
    return dm
end
# Beta Version for Assembling EFG Matrices and Vectors
function AssembleEFG(
    Measures::Union{DomainMeasure, AbstractVector{<:DomainMeasure}},
    Shape_Functions::EFGSpace,
    matrix_type::String; prop=1.0
)
    measures_list = isa(Measures, DomainMeasure) ? [Measures] : Measures
    nnodes = Shape_Functions.nnodes

    all_gs   = Matrix{Float64}(undef, 0, size(first(measures_list).gs, 2))
    all_PHI  = Vector{Vector{Float64}}()
    all_DPHI = Vector{Matrix{Float64}}()
    all_DOM  = Vector{Vector{Int}}()

    for m in measures_list
        tag, gs = m.tag, m.gs

        shape = if haskey(Shape_Functions.domain, tag)
            Shape_Functions.domain[tag]
        elseif haskey(Shape_Functions.boundary, tag)
            Shape_Functions.boundary[tag]
        else
            error("There are no shape functions for the tag '$tag'.")
        end

        PHI, DPHI, DOM = shape

        all_gs   = vcat(all_gs, gs)
        append!(all_PHI,  PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM,  DOM)
    end

    # liberar referencias
    PHI = nothing; DPHI = nothing; DOM = nothing; gs = nothing

    if matrix_type == "Laplacian"
        return COND_MATRIX(prop, all_gs, all_DPHI, all_DOM, nnodes)
    elseif matrix_type == "Mass"
        return CAP_MATRIX(prop, all_gs, all_PHI, all_DOM, nnodes)
    elseif matrix_type == "Load"
        return LOAD_VECTOR(prop, all_gs, all_PHI, all_DOM, nnodes)
    else
        error("Matrix Type '$matrix_type' not recognised. Use 'Laplacian', 'Mass' or 'Load'.")
    end
end

struct EFGMeasure
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Matrix{Float64}}
    DOM::Vector{Vector{Int}}
    nnodes::Int
end
function EFG_Measure(measures::Union{DomainMeasure, AbstractVector{<:DomainMeasure}}, Shape_Functions::EFGSpace)
    # Asegurar que measures sea un vector
    measures_list = isa(measures, DomainMeasure) ? [measures] : measures

    all_PHI = Vector{Vector{Float64}}()
    all_DPHI = Vector{Matrix{Float64}}()
    all_DOM = Vector{Vector{Int}}()

    for sm in measures_list
        tag = sm.tag   # extraer tag del DomainMeasure

        shape = 
            if haskey(Shape_Functions.domain, tag)
                Shape_Functions.domain[tag]
            elseif haskey(Shape_Functions.boundary, tag)
                Shape_Functions.boundary[tag]
            else
                error("No shape functions for tag '$tag'")
            end
        
        PHI, DPHI, DOM = shape
        append!(all_PHI, PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM, DOM)
    end

    # Limpiar referencias temporales
    PHI=DPHI=DOM=nothing
    nnodes = Shape_Functions.nnodes
    return EFGMeasure(all_PHI, all_DPHI, all_DOM, nnodes)
end

include("Fields_Operations.jl")
include("Assembling_Operators.jl")
end