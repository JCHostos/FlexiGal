include("Meshing.jl")
# ------------------------------
# Structs for model
# ------------------------------
struct EFGmodel
    x::Matrix{Float64}                               # Nodes Coordinates
    conn::Matrix{Int}                                # Connectivity
    entities::Dict{Symbol,Dict{String,Union{Int,Matrix{Int}}}} # entidades con tags
    dim::Int
    ncells::Int
    nnodes::Int
end

# ------------------------------
# Function to create a structured cartesian mesh model with tagged entities
# ------------------------------
function create_model(domain::NTuple{D,Float64}, divisions::NTuple{D,Int}) where D
    @assert D == 2 || D == 3 "Solo 2D o 3D soportados"

    # Generar nodos y conectividad
    x, conn, ncells, nnodes = generate_cartesian_mesh(domain, divisions)
    tol = 1e-15
    entities = Dict{Symbol,Dict{String,Union{Int,Matrix{Int}}}}()
    if D == 2
        Nx, Ny = divisions
        Lx, Ly = domain

        # Finding Internal and External edges
        connintedges, connextedges = FindEdgesQuad(conn)
        # ------------------------------
        # Defining Entities
        # ------------------------------
        corner1 = 1
        corner2 = Nx + 1
        corner3 = (Nx + 1) * (Ny) + 1
        corner4 = (Nx + 1) * (Ny + 1)
        # 1) Nodos de las esquinas
        corner_nodes = Dict(
            "Corner1" => corner1,
            "Corner2" => corner2,
            "Corner3" => corner3,
            "Corner4" => corner4
        )
        entities[:nodes] = corner_nodes
        # Tags for edges
        left_edges = Matrix{Int}(undef, Ny, 2)
        right_edges = Matrix{Int}(undef, Ny, 2)
        bottom_edges = Matrix{Int}(undef, Nx, 2)
        top_edges = Matrix{Int}(undef, Nx, 2)
        li = ri = bi = ti = 1
        nedges = size(connextedges, 1)
        @inbounds for e in 1:nedges
            n1, n2 = connextedges[e, 1], connextedges[e, 2]
            xm = (x[n1, 1] + x[n2, 1]) * 0.5
            ym = (x[n1, 2] + x[n2, 2]) * 0.5
            if xm < tol
                left_edges[li, :] .= (n1, n2)
                li += 1
            elseif xm > Lx - tol
                right_edges[ri, :] .= (n1, n2)
                ri += 1
            elseif ym < tol
                bottom_edges[bi, :] .= (n1, n2)
                bi += 1
            elseif ym > Ly - tol
                top_edges[ti, :] .= (n1, n2)
                ti += 1
            end
        end
        ext_edges = Dict(
            "Left" => left_edges,
            "Right" => right_edges,
            "Bottom" => bottom_edges,
            "Top" => top_edges,
            "Boundary" => connextedges
        )
        entities[:ext_edges] = ext_edges
        int_edges = Dict("Internal_Edges" => connintedges)
        entities[:int_edges] = int_edges
        # 3) Faces (2D Cells)
        faces_dict = Dict("Domain" => conn)
        entities[:faces] = faces_dict
    elseif D == 3
        Nx, Ny, Nz = divisions
        Lx, Ly, Lz = domain
        # Finding Internal and External faces and edges
        connintfaces, connextfaces, connintedges, connextedges = FindFacesandEdgesHexa(conn)
        # ------------------------------
        # Defining Entities
        # ------------------------------
        corner1 = 1
        corner2 = Nx + 1
        corner3 = (Nx + 1) * (Ny) + 1
        corner4 = (Nx + 1) * (Ny + 1)
        corner5 = (Nx + 1) * (Ny + 1) * (Nz) + 1
        corner6 = (Nx + 1) * (Ny + 1) * (Nz) + (Nx + 1)
        corner7 = (Nx + 1) * (Ny + 1) * (Nz) + (Nx + 1) * (Ny) + 1
        corner8 = (Nx + 1) * (Ny + 1) * (Nz + 1)
        # 1) Corner Nodes
        corner_nodes = Dict(
            "Corner1" => corner1,
            "Corner2" => corner2,
            "Corner3" => corner3,
            "Corner4" => corner4,
            "Corner5" => corner5,
            "Corner6" => corner6,
            "Corner7" => corner7,
            "Corner8" => corner8
        )
        entities[:nodes] = corner_nodes
        # 2) Tags for External Faces
        left_faces = Matrix{Int}(undef, Nz * Ny, 4)
        right_faces = Matrix{Int}(undef, Nz * Ny, 4)
        bottom_faces = Matrix{Int}(undef, Nx * Ny, 4)
        top_faces = Matrix{Int}(undef, Nx * Ny, 4)
        back_faces = Matrix{Int}(undef, Nx * Nz, 4)
        front_faces = Matrix{Int}(undef, Nx * Nz, 4)
        li = ri = bi = ti = bai = fi = 1
        nfaces = size(connextfaces, 1)
        @inbounds for e in 1:nfaces
            n1, n2, n3, n4 = connextfaces[e, 1], connextfaces[e, 2], connextfaces[e, 3], connextfaces[e, 4]
            xm = (x[n1, 1] + x[n2, 1] + x[n3, 1] + x[n4, 1]) * 0.25
            ym = (x[n1, 2] + x[n2, 2] + x[n3, 2] + x[n4, 2]) * 0.25
            zm = (x[n1, 3] + x[n2, 3] + x[n3, 3] + x[n4, 3]) * 0.25
            if xm < tol
                left_faces[li, :] .= (n1, n2, n3, n4)
                li += 1
            elseif xm > Lx - tol
                right_faces[ri, :] .= (n1, n2, n3, n4)
                ri += 1
            elseif zm < tol
                bottom_faces[bi, :] .= (n1, n2, n3, n4)
                bi += 1
            elseif zm > Lz - tol
                top_faces[ti, :] .= (n1, n2, n3, n4)
                ti += 1
            elseif ym < tol
                back_faces[bai, :] .= (n1, n2, n3, n4)
                bai += 1
            elseif ym > Ly - tol
                front_faces[fi, :] .= (n1, n2, n3, n4)
                fi += 1
            end
        end
        ext_faces = Dict(
            "Left" => left_faces,
            "Right" => right_faces,
            "Bottom" => bottom_faces,
            "Top" => top_faces,
            "Back" => back_faces,
            "Front" => front_faces,
            "Boundary" => connextfaces
        )
        entities[:ext_faces] = ext_faces
        int_faces = Dict("Internal_Faces" => connintfaces)
        entities[:int_faces] = int_faces
        # 2) Tags for External Edges
        nedges = size(connextedges, 1)
        leftbottom_edges = Matrix{Int}(undef, Ny, 2)
        rightbottom_edges = Matrix{Int}(undef, Ny, 2)
        lefttop_edges = Matrix{Int}(undef, Ny, 2)
        righttop_edges = Matrix{Int}(undef, Ny, 2)
        leftback_edges = Matrix{Int}(undef, Nz, 2)
        rightback_edges = Matrix{Int}(undef, Nz, 2)
        leftfront_edges = Matrix{Int}(undef, Nz, 2)
        rightfront_edges = Matrix{Int}(undef, Nz, 2)
        bottomback_edges = Matrix{Int}(undef, Nx, 2)
        bottomfront_edges = Matrix{Int}(undef, Nx, 2)
        toptback_edges = Matrix{Int}(undef, Nx, 2)
        topfront_edges = Matrix{Int}(undef, Nx, 2)
        lbi = rbi = lti = rti = lbai = rbai = lfi = rfi = bbai = bfi = tbai = tfi = 1
        @inbounds for e in 1:nedges
            n1, n2 = connextedges[e, 1], connextedges[e, 2]
            xm = (x[n1, 1] + x[n2, 1]) * 0.5
            ym = (x[n1, 2] + x[n2, 2]) * 0.5
            zm = (x[n1, 3] + x[n2, 3]) * 0.5
            if xm < tol && zm < tol
                leftbottom_edges[lbi, :] .= (n1, n2)
                lbi += 1
            elseif xm > Lx - tol && zm < tol
                rightbottom_edges[rbi, :] .= (n1, n2)
                rbi += 1
            elseif xm < tol && zm > Lz - tol
                lefttop_edges[lti, :] .= (n1, n2)
                lti += 1
            elseif xm > Lx - tol && zm > Lz - tol
                righttop_edges[rti, :] .= (n1, n2)
                rti += 1
            elseif xm < tol && ym < tol
                leftback_edges[lbai, :] .= (n1, n2)
                lbai += 1
            elseif xm > Lx - tol && ym < tol
                rightback_edges[rbai, :] .= (n1, n2)
                rbai += 1
            elseif xm < tol && ym > Ly - tol
                leftfront_edges[lfi, :] .= (n1, n2)
                lfi += 1
            elseif xm > Lx - tol && ym > Ly - tol
                rightfront_edges[rfi, :] .= (n1, n2)
                rfi += 1
            elseif zm < tol && ym < tol
                bottomback_edges[bbai, :] .= (n1, n2)
                bbai += 1
            elseif zm < tol && ym > Ly - tol
                bottomfront_edges[bfi, :] .= (n1, n2)
                bfi += 1
            elseif zm > Lz - tol && ym < tol
                toptback_edges[tbai, :] .= (n1, n2)
                tbai += 1
            elseif zm > Lz - tol && ym > Ly - tol
                topfront_edges[tfi, :] .= (n1, n2)
                tfi += 1
            end
        end
        ext_edges = Dict(
            "leftbottom" => leftbottom_edges,
            "rightbottom" => rightbottom_edges,
            "lefttop" => lefttop_edges,
            "righttop" => righttop_edges,
            "leftback" => leftback_edges,
            "rightback" => rightback_edges,
            "leftfront" => leftfront_edges,
            "rightfront" => rightfront_edges,
            "bottomback" => bottomback_edges,
            "bottomfront" => bottomfront_edges,
            "topback" => toptback_edges,
            "topfront" => topfront_edges,
            "External_Edges" => connextedges
        )
        entities[:ext_edges] = ext_edges
        int_edges = Dict("Internal_Edges" => connintedges)
        entities[:int_edges] = int_edges
        # 3) Volumes (3D Cells)
        faces_dict = Dict("Domain" => conn)
        entities[:volumes] = faces_dict
    end
    # Caras (faces) temporales
    connintfaces = nothing
    connextfaces = nothing
    left_faces = nothing
    right_faces = nothing
    bottom_faces = nothing
    top_faces = nothing
    back_faces = nothing
    front_faces = nothing

    # Aristas (edges) temporales
    connintedges = nothing
    connextedges = nothing
    leftbottom_edges = nothing
    rightbottom_edges = nothing
    lefttop_edges = nothing
    righttop_edges = nothing
    leftback_edges = nothing
    rightback_edges = nothing
    leftfront_edges = nothing
    rightfront_edges = nothing
    bottomback_edges = nothing
    bottomfront_edges = nothing
    toptback_edges = nothing
    topfront_edges = nothing
    GC.gc()
    # ------------------------------
    # Return model
    # ------------------------------
    return EFGmodel(x, conn, entities, D, ncells, nnodes)
end

function get_entity(model::EFGmodel, tag::String)
    for ent_type in keys(model.entities)
        subdict = get(model.entities, ent_type, Dict{String,Union{Int,Matrix{Int}}}())
        if haskey(subdict, tag)
            return subdict[tag]
        end
    end
    error("Tag '$tag' at any entity of the model.")
end