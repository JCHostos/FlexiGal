module FlexiGal
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
export create_model, BackgroundIntegration, EFGSpace, Influence_Domains, AssembleEFG, EFGFunction,
    Domain_Measure, Get_Point_Values, ∇, Internal_Product, Integrate, ∫, ⋅, *, EFG_Measure, Bilinear_Assembler


function BackgroundIntegration(model::EFGmodel, tag::String, degree::Int)
    conn = get_entity(model, tag)
    gauss = pgauss(degree)
    dim = size(model.x, 2)
    if size(conn, 2) == 2^dim
        gs = egauss(model.x, conn, gauss)
    elseif size(conn, 2) == 2^(dim - 1)
        gs = egauss_bound(model.x, conn, gauss)
    end
    return (tag, gs)
end
# Constructing Shape Functions given model, Measures for Integration and Influence domains
struct EFGSpace
    domain::Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}
    boundary::Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}
    nnodes::Int
end
function EFGSpace(model::EFGmodel,
                  gs_list::Union{Tuple{String,Matrix{Float64}}, Vector{Tuple{String,Matrix{Float64}}}},
                  dm::Matrix{Float64})
    # Forzar que gs_list siempre sea un vector
    gs_list = isa(gs_list, Tuple) ? [gs_list] : gs_list

    x = model.x
    results_domain = Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}()
    results_boundary = Dict{String,Tuple{Vector{Vector{Float64}},Vector{Matrix{Float64}},Vector{Vector{Int}}}}()
    nnodes, dim = size(x)

    for (tag, gs) in gs_list
        conn = get_entity(model, tag)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary

        PHI, DPHI, DOM = SHAPE_FUN(gs, x, dm)

        if gs_type == :domain
            results_domain[tag] = (PHI, DPHI, DOM)
        else
            results_boundary[tag] = (PHI, DPHI, DOM)
        end
    end

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
function AssembleEFG(Measures::Union{Tuple{String,Matrix{Float64}},
        AbstractVector{<:Tuple{String,Matrix{Float64}}}},
    Shape_Functions::EFGSpace,
    matrix_type::String; prop=1.0)
    measures_list = isa(Measures, Tuple) ? [Measures] : Measures
    nnodes = Shape_Functions.nnodes
    all_gs = Matrix{Float64}(undef, 0, size(first(measures_list)[2], 2))
    all_PHI = Vector{Vector{Float64}}()
    all_DPHI = Vector{Matrix{Float64}}()
    all_DOM = Vector{Vector{Int}}()

    for (tag, gs) in measures_list
        shape = if haskey(Shape_Functions.domain, tag)
            Shape_Functions.domain[tag]
        elseif haskey(Shape_Functions.boundary, tag)
            Shape_Functions.boundary[tag]
        else
            error("There are no shape functions for the tag '$tag'.")
        end

        PHI, DPHI, DOM = shape

        all_gs = vcat(all_gs, gs)
        append!(all_PHI, PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM, DOM)
    end

    if matrix_type == "Laplacian"
        return COND_MATRIX(prop, all_gs, all_DPHI, all_DOM, nnodes)
    elseif matrix_type == "Mass"
        return CAP_MATRIX(prop, all_gs, all_PHI, all_DOM, nnodes)
    elseif matrix_type == "Load"
        return LOAD_VECTOR(prop, all_gs, all_PHI, all_DOM, nnodes)
    else
        error("Matrix Type '$matrix_type' no recognised. Please use 'Laplacian', 'Mass' or 'Load'.")
    end
end

struct EFGMeasure
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Matrix{Float64}}
    DOM::Vector{Vector{Int}}
    gs::Matrix{Float64}
    nnodes::Int
end
function EFG_Measure(Measures::Union{Tuple{String,Matrix{Float64}},
        AbstractVector{<:Tuple{String,Matrix{Float64}}}}, Shape_Functions::EFGSpace)
    measures_list = isa(Measures, Tuple) ? [Measures] : Measures
    all_gs = Matrix{Float64}(undef, 0, size(first(measures_list)[2], 2))
    all_PHI = Vector{Vector{Float64}}()
    all_DPHI = Vector{Matrix{Float64}}()
    all_DOM = Vector{Vector{Int}}()
    for (tag, gs) in measures_list # Buscar en domain o boundary 
        shape =
            if haskey(Shape_Functions.domain, tag)
                Shape_Functions.domain[tag]
            elseif haskey(Shape_Functions.boundary, tag)
                Shape_Functions.boundary[tag]
            else
                error("There are no shape functions for the tag '$tag'.")
            end
        PHI, DPHI, DOM = shape
        all_gs = vcat(all_gs, gs)
        append!(all_PHI, PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM, DOM)
    end
    nnodes = Shape_Functions.nnodes
    return EFGMeasure(all_PHI, all_DPHI, all_DOM, all_gs, nnodes)
end
include("Fields_Operations.jl")
include("Assembling_Operators.jl")
end