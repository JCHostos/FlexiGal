module FlexiGal
using StaticArrays
using LinearAlgebra
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
include("WeakForm_Macro.jl")
export create_model, BackgroundIntegration, EFG_Space, Influence_Domains, AssembleEFG, EFGFunction, ApproxSpace, Solve, build_space,
    Domain_Measure, Get_Point_Values, ∇, Internal_Product, Integrate, ∫, ⋅, ⊙, Bilinear_Assembler, Linear_Assembler, VectorField, Linear_Problem, plot_field,
    Triangulation, IntegrationSet, Get_space_from_IntegrationSet, tr, Id, Prueba_Macro, @WeakForm, @NL_WeakForm, NonLinearOperator, Get_Nodal_Values, Get_Measures
struct Triangulation
    model::EFGmodel
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
function BackgroundIntegration(iset::IntegrationSet)
    model = iset.tri.model
    tag = iset.tri.tag
    degree = iset.degree
    conn = get_entity(model, tag)
    gauss = pgauss(degree)
    dim = size(model.x, 2)
    if size(conn, 2) == 2^dim
        gs = egauss(model.x, conn, gauss)
    elseif size(conn, 2) == 2^(dim - 1)
        gs = egauss_bound(model.x, conn, gauss)
    end
    conn = nothing
    return DomainMeasure(tag, gs,degree)
end
@inline merge(measure::DomainMeasure; tag="merged") = measure

@inline function merge(measures::Vector{DomainMeasure}; tag::String="merged")
    isempty(measures) && error("No DomainMeasure to merge")
    all_gs = vcat([m.gs for m in measures]...)
    return DomainMeasure(tag, all_gs, measures[1].degree)
end

struct ApproxSpace{DV}
    model::EFGmodel
    triangulations::Vector{Triangulation} # Antes era measures
    Field_Type::Type
    dmax::Union{Float64,Vector{Float64}}
    Dirichlet_Boundaries::Vector{Triangulation}
    Dirichlet_Values::Vector{DV}
end

# El constructor que usaremos en tu script
function ApproxSpace(model, triangs, Field_Type, dmax;
    Dirichlet_Boundaries = Triangulation[],
    Dirichlet_Values = Float64[])
    T = triangs isa Triangulation ? [triangs] : triangs
    DT = Dirichlet_Boundaries isa Triangulation ? [Dirichlet_Boundaries] : Dirichlet_Boundaries
    DV = eltype(Dirichlet_Values) 
    return ApproxSpace{DV}(model, T, Field_Type, dmax, DT, Dirichlet_Values)
end
# Constructing Shape Functions given model, Measures for Integration and Influence domains
struct EFGSpace{DG}
    domain::Dict{String,Tuple{
        Vector{Vector{Float64}},
        Vector{Vector{SVector{DG, Float64}}},
        Vector{Vector{Int}}
    }}
    boundary::Dict{String,Tuple{
        Vector{Vector{Float64}},
        Vector{Vector{SVector{DG, Float64}}},
        Vector{Vector{Int}}
    }}
    Field_Type::Type
    Measures::Vector{DomainMeasure}
    nnodes::Int
end

function EFG_Space(model::EFGmodel{DG}, # DG sale del modelo (2 o 3)
    gs_list::Vector{DomainMeasure},
    Field_Type::Type,
    dm::Matrix{Float64}) where DG

    x = model.x
    
    # Definimos el tipo de la tupla para no escribirlo mil veces
    T_Shape = Tuple{Vector{Vector{Float64}}, Vector{Vector{SVector{DG, Float64}}}, Vector{Vector{Int}}}
    
    results_domain = Dict{String, T_Shape}()
    results_boundary = Dict{String, T_Shape}()
    
    nnodes, dim = size(x)
    
    for measure in gs_list
        tag = measure.tag
        gs = measure.gs
        conn = get_entity(model, tag)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary
        
        # Ahora SHAPE_FUN devuelve los SVectors perfectamente
        PHI, DPHI, DOM = SHAPE_FUN(gs, x, dm)
        
        if gs_type === :domain
            results_domain[tag] = (PHI, DPHI, DOM)
        else
            results_boundary[tag] = (PHI, DPHI, DOM)
        end
    end
    
    return EFGSpace{DG}( # Pasamos el parámetro DG
        results_domain,
        results_boundary,
        Field_Type,
        gs_list,
        nnodes
    )
end
# Defining Influence Domains (Ongoing Development)
function Influence_Domains(model::EFGmodel, Domain::Tuple, Divisions::Tuple, dmax::Union{Real,AbstractVector{<:Real}})
    dim = size(model.x, 2)
    dm = zeros(size(model.x, 1), dim)
    # Convertir dmax a vector de longitud dim
    dvec = isa(dmax, Real) ? fill(dmax, dim) : dmax
    @assert length(dvec) == dim "Length of dmax vector must match problem dimension"
    # Extraer dimensiones y divisiones
    if dim == 2
        Lx, Ly = Domain
        Nx, Ny = Divisions
        dm[:, 1] .= dvec[1] * Lx / Nx
        dm[:, 2] .= dvec[2] * Ly / Ny
    elseif dim == 3
        Lx, Ly, Lz = Domain
        Nx, Ny, Nz = Divisions
        dm[:, 1] .= dvec[1] * Lx / Nx
        dm[:, 2] .= dvec[2] * Ly / Ny
        dm[:, 3] .= dvec[3] * Lz / Nz
    else
        error("Only 2D and 3D supported")
    end
    return dm
end

struct EFGMeasure{DG}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Vector{SVector{DG,Float64}}} # Formato optimizado
    DOM::Vector{Vector{Int}}
    nnodes::Int
end
function EFG_Measure(measures::Union{DomainMeasure,AbstractVector{<:DomainMeasure}}, Shape_Functions::EFGSpace)
    measures_list = isa(measures, DomainMeasure) ? [measures] : measures
    DG = typeof(Shape_Functions).parameters[1] 
    all_PHI = Vector{Vector{Float64}}()
    all_DPHI = Vector{Vector{SVector{DG, Float64}}}() 
    all_DOM = Vector{Vector{Int}}()
    for sm in measures_list
        tag = sm.tag
        shape = haskey(Shape_Functions.domain, tag) ? Shape_Functions.domain[tag] :
                haskey(Shape_Functions.boundary, tag) ? Shape_Functions.boundary[tag] :
                error("No shape functions for tag '$tag'")
        PHI, DPHI, DOM = shape 
        append!(all_PHI, PHI)
        append!(all_DPHI, DPHI) # Ahora append! solo copia punteros a los SVectors
        append!(all_DOM, DOM)
    end
    return EFGMeasure{DG}(all_PHI, all_DPHI, all_DOM, Shape_Functions.nnodes)
end

function build_space(recipe::ApproxSpace, isets::Vector{IntegrationSet})
    m  = recipe.model
    dm = Influence_Domains(m, m.domain, m.divisions, recipe.dmax)
    Measures = [BackgroundIntegration(iset) for iset in isets]
    return EFG_Space(
        m,
        Measures,
        recipe.Field_Type,
        dm
    )
end

include("Fields_Operations.jl")
include("Assembling_Operators.jl")
include("Solvers.jl")
include("Plots.jl")
end