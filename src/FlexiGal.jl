module FlexiGal
using StaticArrays
using LinearAlgebra
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
include("WeakForm_Macro.jl")
export create_model, BackgroundIntegration, Flexi_Space, Influence_Domains, FlexiFunction, ApproxSpace, Solve, build_space,
    Domain_Measure, Get_Point_Values, ∇, Internal_Product, Integrate, ∫, ⋅, ⊙, Bilinear_Assembler, Linear_Assembler, VectorField, Linear_Problem, plot_field,
    Triangulation, IntegrationSet, Get_space_from_IntegrationSet, tr, Id, NL_Solver, @WeakForm, @NL_WeakForm, NonLinearOperator, Get_Nodal_Values, Get_Measures,
    Reassemble_Vector!, set_tag!
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

function BackgroundIntegration(iset::IntegrationSet, method::Symbol)
    model = iset.tri.model
    tag = iset.tri.tag
    degree = iset.degree
    dim = model.dim
    conn, cat = get_entity_info(model, tag)
    gauss = pgauss(degree)
    domain_cat = (dim == 2) ? :faces : :volumes
    is_boundary = (cat !== domain_cat)
    if method === :FEM
        if !is_boundary
            gs, PHI, DPHI, DOM = egauss_fem(model.x, conn, gauss)
        else
            gs, PHI, DPHI, DOM = egauss_bound_fem(model.x, conn, gauss)
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
    Dirichlet_Boundaries::Vector{Triangulation}
    Dirichlet_Values::Vector{DV}
end

# El constructor que usaremos en tu script
function ApproxSpace(model, triangs, Field_Type; dmax=1.5, shape::Symbol=:rectangular, technique=:IMLS,
    method=:EFG, dm_map=nothing,
    Dirichlet_Boundaries=Triangulation[],
    Dirichlet_Values=Float64[])
    T = triangs isa Triangulation ? [triangs] : triangs
    DT = Dirichlet_Boundaries isa Triangulation ? [Dirichlet_Boundaries] : Dirichlet_Boundaries
    DV = eltype(Dirichlet_Values)
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
    return ApproxSpace{DV}(model, T, Field_Type, dmax, shape, technique, methods_dict,dm_map, DT, Dirichlet_Values)
end

# Constructing Shape Functions given model, Measures for Integration and Influence domains
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

function Flexi_Space(model::model{DG}, gs_list::Vector{DomainMeasure}, Field_Type::Type, dm::Union{Matrix{Float64},Vector{Float64}}; shape::Symbol=:rectangular, technique::Symbol=:IMLS) where DG
    x = model.x
    T_Shape = Tuple{Vector{Vector{Float64}},Vector{Vector{SVector{DG,Float64}}},Vector{Vector{Int}}}
    results_domain = Dict{String,T_Shape}()
    results_boundary = Dict{String,T_Shape}()
    nnodes, dim = size(x)
    for measure in gs_list
        tag = measure.tag
        gs = measure.gs
        conn = get_entity(model, tag)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary
        PHI, DPHI, DOM = SHAPE_FUN(gs, x, dm; shape=shape, technique=technique)
        if gs_type === :domain
            results_domain[tag] = (PHI, DPHI, DOM)
        else
            results_boundary[tag] = (PHI, DPHI, DOM)
        end
    end
    return FlexiSpace{DG}(results_domain, results_boundary, Field_Type, gs_list, nnodes)
end

function Flexi_Space(model::model{DG}, gs_list::Vector{DomainMeasure}, Field_Type::Type, results::Vector) where DG
    x = model.x
    T_Shape = Tuple{Vector{Vector{Float64}},Vector{Vector{SVector{DG,Float64}}},Vector{Vector{Int}}}
    results_domain = Dict{String,T_Shape}()
    results_boundary = Dict{String,T_Shape}()
    nnodes, dim = size(x)
    for res in results
        measure, PHI, DPHI, DOM = res
        tag = measure.tag
        conn = get_entity(model, tag)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary
        if gs_type === :domain
            results_domain[tag] = (PHI, DPHI, DOM)
        else
            results_boundary[tag] = (PHI, DPHI, DOM)
        end
    end
    return FlexiSpace{DG}(results_domain, results_boundary, Field_Type, gs_list, nnodes)
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

function build_space(recipe::ApproxSpace, isets::Vector{IntegrationSet})
    m = recipe.model
    dim = size(m.x, 2)
    DG = dim
    T_Shape = Tuple{Vector{Vector{Float64}},Vector{Vector{SVector{DG,Float64}}},Vector{Vector{Int}}}
    results_domain = Dict{String,T_Shape}()
    results_boundary = Dict{String,T_Shape}()
    gs_list = DomainMeasure[]
    shape = recipe.shape
    technique = recipe.technique
    dm_map = recipe.dm_map
    needs_efg = any(get(recipe.methods, iset.tri.tag, :EFG) === :EFG for iset in isets)
    dm = needs_efg ? Influence_Domains(m, m.domain, m.divisions, recipe.dmax; shape=shape, dm_map=dm_map) : Matrix{Float64}(undef, 0, 0)
    for iset in isets
        tag = iset.tri.tag
        method = get(recipe.methods, tag, :EFG)
        conn, cat = get_entity_info(m, tag)
        domain_cat = (dim == 2) ? :faces : :volumes
        gs_type = (cat === domain_cat) ? :domain : :boundary
        if method === :FEM
            meas, PHI, DPHI, DOM = BackgroundIntegration(iset, :FEM)
        else
            meas = BackgroundIntegration(iset, :EFG)
            if gs_type === :domain
                global_node_ids = sort!(unique(view(conn, :)))
                PHI, DPHI, DOM_local = SHAPE_FUN(meas.gs, m.x[global_node_ids,:], dm[global_node_ids,:]; shape=shape, technique=technique)
                DOM = [global_node_ids[idx] for idx in DOM_local]
            else
                PHI, DPHI, DOM = SHAPE_FUN(meas.gs, m.x, dm; shape=shape, technique=technique)
            end
        end
        push!(gs_list, meas)
        target_dict = (gs_type === :domain) ? results_domain : results_boundary
        target_dict[tag] = (PHI, DPHI, DOM)
    end
    return FlexiSpace{DG}(results_domain, results_boundary, recipe.Field_Type, gs_list, m.nnodes)
end

include("Fields_Operations.jl")
include("Assembling_Operators.jl")
include("Solvers.jl")
include("Plots.jl")
end