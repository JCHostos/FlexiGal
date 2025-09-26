using SparseArrays
function COND_MATRIX(k, gs::Matrix{Float64}, DPHI::Vector{Matrix{Float64}}, DOM::Vector{Vector{Int}}, nnod::Int)
    numqc = length(DOM)
    dim = size(DPHI[1],2)
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        w = gs[ind, end]
        detJ = gs[ind, end-1]
        dom = DOM[ind]                 # nodos de influencia
        nvec = length(dom)
        # Inicializar contenedores para este punto
        row[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)           # vector de largo nvec^2
        col[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)           # idem
        Kloc = Array{Float64}(undef, nvec, nvec) # matriz local
        # derivadas en x e y (dim × nvec)
        if dim == 2
            DPHIx = DPHI[ind][:, 1]
            DPHIy = DPHI[ind][:, 2]
        elseif dim == 3
            DPHIx = DPHI[ind][:, 1]
            DPHIy = DPHI[ind][:, 2]
            DPHIz = DPHI[ind][:, 3]
        else
            error("Dimension not supported")
        end
        # Rellenar matriz local
        if dim == 2
            @inbounds for a in 1:nvec
                @simd for b in 1:nvec
                    Kloc[a, b] = (DPHIx[a] * k * DPHIx[b] + DPHIy[a] * k * DPHIy[b]) * w * detJ
                end
            end
        elseif dim == 3
            @inbounds for a in 1:nvec
                @simd for b in 1:nvec
                    Kloc[a, b] = (DPHIx[a] * k * DPHIx[b] + DPHIy[a] * k * DPHIy[b] + DPHIz[a] * k * DPHIz[b]) * w * detJ
                end
            end
        end
        val[pos:pos+nvec^2-1] = vec(Kloc)
        pos += nvec^2
    end
    K = sparse(row, col, val, nnod, nnod)
    K = 0.5 * (K' + K)  # simetrizar
    return K
end

function CAP_MATRIX(ρc, gs::Matrix{Float64}, PHI::Vector{Vector{Float64}}, DOM::Vector{Vector{Int}}, nnod::Int)
    numqc = length(DOM)
    # Pre-alocar vectores de vectores/matrices
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        w = gs[ind, end]
        detJ = gs[ind, end-1]
        dom = DOM[ind]                 # nodos de influencia
        nvec = length(dom)
        phi = PHI[ind]
        # Inicializar contenedores para este punto
        row[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)           # vector de largo nvec^2
        col[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)           # idem
        Cloc = Array{Float64}(undef, nvec, nvec) # matriz local
        # Rellenar matriz local
        @inbounds for a in 1:nvec
            @simd for b in 1:nvec
                Cloc[a, b] = (phi[a] * ρc * phi[b]) * w * detJ
            end
        end
        val[pos:pos+nvec^2-1] = vec(Cloc)
        pos += nvec^2
    end
    C = sparse(row, col, val, nnod, nnod)
    C = 0.5 * (C' + C)  # simetrizar
    return C
end

function LOAD_VECTOR(dd, gs, PHI, DOM, nnod)
    numqc = length(DOM)
    total_length = sum(length.(DOM))            # nº de puntos de Gauss
    row = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1         # contribuciones locales
    @inbounds for ind in 1:numqc
        n = length(DOM[ind])
        w = gs[ind, end]      # penúltima columna = peso
        detJ = gs[ind, end-1]        # última columna     = det
        val[pos:pos+n-1] = PHI[ind] * dd * detJ * w
        row[pos:pos+n-1] = DOM[ind]
         pos += n
    end
    # construir vector disperso global (nnod × 1)
    Qpf = vec(Array(sparse(row, ones(Int, length(row)), val, nnod, 1)))
    return Qpf
end