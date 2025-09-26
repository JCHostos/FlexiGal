function generate_cartesian_mesh(domain::NTuple{D,Float64}, divisions::NTuple{D,Int}) where D
    @assert D == 2 || D == 3 "Only 2D and 3D meshes are supported"

    nnodes = prod(d + 1 for d in divisions)
    ncells = prod(divisions)

    x = zeros(Float64, nnodes, D)

    if D == 2
        Nx, Ny = divisions
        Lx, Ly = domain
        for j in 0:Ny
            for i in 0:Nx
                idx = i + 1 + (Nx + 1) * j
                x[idx, :] = [(Lx / Nx) * i, (Ly / Ny) * j]
            end
        end
        conn = zeros(Int, ncells, 4)
        for j in 0:(Ny-1)
            for i in 0:(Nx-1)
                e = i + 1 + j * Nx
                n1 = i + 1 + j * (Nx + 1)
                n2 = n1 + 1
                n3 = n2 + Nx + 1
                n4 = n3 - 1
                conn[e, :] = [n1, n2, n3, n4]
            end
        end
    else  # D == 3
        Nx, Ny, Nz = divisions
        Lx, Ly, Lz = domain
        for k in 0:Nz
            for j in 0:Ny
                for i in 0:Nx
                    idx = i + 1 + (Nx + 1) * j + (Nx + 1) * (Ny + 1) * k
                    x[idx, :] = [(Lx / Nx) * i, (Ly / Ny) * j, (Lz / Nz) * k]
                end
            end
        end
        conn = zeros(Int, ncells, 8)
        for k in 0:(Nz-1)
            for j in 0:(Ny-1)
                for i in 0:(Nx-1)
                    e = i + j * Nx + k * Nx * Ny + 1
                    n0 = i + 1 + j * (Nx + 1) + k * (Nx + 1) * (Ny + 1)
                    n1 = n0 + 1
                    n2 = n1 + Nx + 1
                    n3 = n0 + Nx + 1
                    n4 = n0 + (Nx + 1) * (Ny + 1)
                    n5 = n1 + (Nx + 1) * (Ny + 1)
                    n6 = n2 + (Nx + 1) * (Ny + 1)
                    n7 = n3 + (Nx + 1) * (Ny + 1)
                    conn[e, :] = [n0, n1, n2, n3, n4, n5, n6, n7]
                end
            end
        end
    end

    return x, conn, ncells, nnodes
end

function Coords_by_Ele(x::Matrix{Float64}, conn::Matrix{Int})
    nele, nnodxele = size(conn)
    dim = size(x, 2)                           # 2 para 2D, 3 para 3D…
    xe = zeros(Float64, nele, nnodxele, dim)   # (elemento, nodo por elemento, coordenada)

    for i in 1:nnodxele
        c = conn[:, i]                         # índices de los nodos para la i-ésima posición
        xe[:, i, :] = x[c, :]                   # copiar las coordenadas correspondientes
    end

    return xe
end
function matching_cols(A, B)
    ext = size(A, 1)
    int = size(B, 1)
    index = fill(0, ext)
    @inbounds for i in 1:ext
        @inbounds for j in 1:int
            if A[i, 1] == B[j, 1] && A[i, 2] == B[j, 2]
                if index[i] == 0
                    index[i] = j
                end
            end
        end
    end
    return index
end
function matching_cols_dict(A::Matrix{Int}, B::Matrix{Int})
    dictB = Dict{NTuple{size(B,2),Int},Int}()          # crear vacío
    sizehint!(dictB, size(B,1))                        # sugerir capacidad
    @inbounds for (i, r) in enumerate(eachrow(B))
        dictB[Tuple(r)] = i
    end
    @inbounds return [dictB[Tuple(r)] for r in eachrow(A)]
end

function conn_external(conn::Matrix{Int}, connint::Matrix{Int})
    nrows=size(conn,2)
    dict_int = Dict{NTuple{nrows,Int},Bool}()
    conn_sorted=sort(conn, dims=2)
    connint_sorted=sort(connint, dims=2)
    @inbounds for r in eachrow(connint_sorted)
        dict_int[Tuple(r)] = true
    end

    nrows = size(conn_sorted,1)
    idx_ext = Vector{Int}(undef, nrows)
    count = 0
    @inbounds for (i,r) in enumerate(eachrow(conn_sorted))
        key = Tuple(r)
        if !haskey(dict_int, key)
            count += 1
            idx_ext[count] = i
        end
    end
    return conn[idx_ext[1:count], :]
end

function FindEdgesQuad(conn::Matrix{Int})
    nelems, _ = size(conn)
    all = Array{Int,2}(undef, nelems*4, 2)
    row = 1
    @inbounds for e in 1:nelems
        all[row, :] = conn[e, 1:2];   row += 1
        all[row, :] = conn[e, 2:3];   row += 1
        all[row, :] = conn[e, 3:4];   row += 1
        all[row, :] = conn[e, [4,1]]; row += 1
    end

    all_sorted = sort(all, dims=2)
    m = unique(all_sorted, dims=1)
    index = matching_cols_dict(m, all_sorted)
    connedges = all[index, :]

    mask = trues(size(all,1))
    mask[index] .= false
    connintedges = all[mask, :]
    connintedges_sorted = sort(connintedges, dims=2)
    m_int = unique(connintedges_sorted, dims=1)
    index_int = matching_cols_dict(m_int, connintedges_sorted)
    connintedges = connintedges[index_int, :]
    # Aristas externas
    connextedges = conn_external(connedges, connintedges)
    return connintedges, connextedges
end

function FindFacesandEdgesHexa(conn::Matrix{Int})
    nelems, _ = size(conn)
    all = Array{Int,2}(undef, nelems*6, 4)
    row = 1
    @inbounds for e in 1:nelems
        all[row, :] = conn[e, [1,4,8,5]];   row += 1
        all[row, :] = conn[e, [2,3,7,6]];   row += 1
        all[row, :] = conn[e, [1,2,6,5]];   row += 1
        all[row, :] = conn[e, [3,7,8,4]];   row += 1
        all[row, :] = conn[e, [1,2,3,4]];   row += 1
        all[row, :] = conn[e, [5,6,7,8]];   row += 1
    end

    all_sorted = sort(all, dims=2)
    m = unique(all_sorted, dims=1)
    index = matching_cols_dict(m, all_sorted)
    connfaces = all[index, :]

    mask = trues(size(all,1))
    mask[index] .= false
    connintfaces = all[mask, :]
    connintfaces_sorted = sort(connintfaces, dims=2)
    m_int = unique(connintfaces_sorted, dims=1)
    index_int = matching_cols_dict(m_int, connintfaces_sorted)
    connintfaces = connintfaces[index_int, :]

    connextfaces = conn_external(connfaces, connintfaces)
    nfaces = size(connintfaces, 1)
    all = Array{Int,2}(undef, nfaces*4, 2)
    row = 1
    @inbounds for e in 1:nfaces
        all[row, :] = connintfaces[e, [1,2]];   row += 1
        all[row, :] = connintfaces[e, [2,3]];   row += 1
        all[row, :] = connintfaces[e, [3,4]];   row += 1
        all[row, :] = connintfaces[e, [4,1]];   row += 1
    end

    all_sorted = sort(all, dims=2)
    m = unique(all_sorted, dims=1)
    index = matching_cols_dict(m, all_sorted)
    connedges = all[index, :]
    mask = trues(size(all,1))
    mask[index] .= false
    connintedges = all[mask, :]
    connintedges_sorted = sort(connintedges, dims=2)
    m_int = unique(connintedges_sorted, dims=1)
    index_int = matching_cols_dict(m_int, connintedges_sorted)
    connintedges = connintedges[index_int, :]
    nfaces = size(connfaces, 1)
    all = Array{Int,2}(undef, nfaces*4, 2)  # preasignamos tamaño
    row = 1
    @inbounds for e in 1:nfaces
        all[row, :] = connfaces[e, [1,2]];   row += 1
        all[row, :] = connfaces[e, [2,3]];   row += 1
        all[row, :] = connfaces[e, [3,4]];   row += 1
        all[row, :] = connfaces[e, [4,1]];   row += 1
    end

    all_sorted = sort(all, dims=2)
    m = unique(all_sorted, dims=1)
    index = matching_cols_dict(m, all_sorted)
    connedges = all[index, :]
    connextedges = conn_external(connedges, connintedges)
    return connintfaces, connextfaces, connintedges, connextedges
end