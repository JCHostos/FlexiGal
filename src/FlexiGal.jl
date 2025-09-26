module FlexiGal
include("Geometry.jl")
include("Shape_Functions.jl")
include("Integration.jl")
include("Assembling_Operators.jl")
export create_model, BackgroundIntegration, EFG_Functions, Influence_Domains, AssembleEFG

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
function EFG_Functions(model::EFGmodel,
    gs_list::Vector{Tuple{String,Matrix{Float64}}},
    dm::Matrix{Float64})
    x = model.x
    results = Dict(:domain => Dict{String,Tuple{Vector{Vector{Float64}},
            Vector{Matrix{Float64}},
            Vector{Vector{Int}}}}(),
        :boundary => Dict{String,Tuple{Vector{Vector{Float64}},
            Vector{Matrix{Float64}},
            Vector{Vector{Int}}}}())

    for (tag, gs) in gs_list
        conn = get_entity(model, tag)
        dim = size(x, 2)
        gs_type = size(conn, 2) == 2^dim ? :domain : :boundary

        PHI, DPHI, DOM = SHAPE_FUN(gs, x, dm)
        results[gs_type][tag] = (PHI, DPHI, DOM)
    end

    return results
end
# Defining Influence Domains (Ongoing Development)
function Influence_Domains(model, Domain, Divisions, dmax)
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
function AssembleEFG(model,
    Measures::Union{Tuple{String,Matrix{Float64}},
                     AbstractVector{<:Tuple{String,Matrix{Float64}}}},
    Shape_Functions::Dict,
    matrix_type::String; prop=1.0)

    measures_list = isa(Measures, Tuple) ? [Measures] : Measures
    nnod = size(model.x, 1)

    all_gs   = Matrix{Float64}(undef, 0, size(first(measures_list)[2],2))
    all_PHI  = Vector{Vector{Float64}}()
    all_DPHI = Vector{Matrix{Float64}}()
    all_DOM  = Vector{Vector{Int}}()

    for (tag, gs) in measures_list
        shape = if haskey(Shape_Functions[:domain], tag)
            Shape_Functions[:domain][tag]
        elseif haskey(Shape_Functions[:boundary], tag)
            Shape_Functions[:boundary][tag]
        else
            error("There are no shape functions for the tag '$tag'.")
        end

        PHI, DPHI, DOM = shape

        all_gs   = vcat(all_gs, gs)
        append!(all_PHI,  PHI)
        append!(all_DPHI, DPHI)
        append!(all_DOM,  DOM)
    end
    if matrix_type == "Laplacian"
        return COND_MATRIX(prop, all_gs, all_DPHI, all_DOM, nnod)
    elseif matrix_type == "Mass"
        return CAP_MATRIX(prop, all_gs, all_PHI, all_DOM, nnod)
    elseif matrix_type == "Load"
        return LOAD_VECTOR(prop, all_gs, all_PHI, all_DOM, nnod)
    else
        error("Matrix Type '$matrix_type' no recognised. Please use 'Laplacian', 'Mass' or 'Load'.")
    end
end
end