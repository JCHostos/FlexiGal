using SparseArrays
function COND_MATRIX(k, gs::Matrix{Float64}, DPHI::Vector{Matrix{Float64}}, DOM::Vector{Vector{Int}}, nnod::Int)
    numqc = length(DOM)
    dim = size(DPHI[1], 2)
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        w = gs[ind, end]
        detJ = gs[ind, end-1]
        dom = DOM[ind]
        nvec = length(dom)
        row[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)
        col[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)
        Kloc = Array{Float64}(undef, nvec, nvec)
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
    return K
end

function CAP_MATRIX(ρc, gs::Matrix{Float64}, PHI::Vector{Vector{Float64}}, DOM::Vector{Vector{Int}}, nnod::Int)
    numqc = length(DOM)
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        w = gs[ind, end]
        detJ = gs[ind, end-1]
        dom = DOM[ind]
        nvec = length(dom)
        phi = PHI[ind]
        row[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)
        col[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)
        Cloc = Array{Float64}(undef, nvec, nvec)
        @inbounds for a in 1:nvec
            @simd for b in 1:nvec
                Cloc[a, b] = (phi[a] * ρc * phi[b]) * w * detJ
            end
        end
        val[pos:pos+nvec^2-1] = vec(Cloc)
        pos += nvec^2
    end
    C = sparse(row, col, val, nnod, nnod)
    return C
end

function LOAD_VECTOR(dd, gs, PHI, DOM, nnod)
    numqc = length(DOM)
    total_length = sum(length.(DOM))
    row = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        n = length(DOM[ind])
        w = gs[ind, end]
        detJ = gs[ind, end-1]
        val[pos:pos+n-1] = PHI[ind] * dd * detJ * w
        row[pos:pos+n-1] = DOM[ind]
        pos += n
    end
    Qpf = vec(Array(sparse(row, ones(Int, length(row)), val, nnod, 1)))
    return Qpf
end
function Bilinear_Assembler(f::Function, Space::EFGSpace)
    _, dX = f(1, 1)
    Shapes = EFG_Measure(dX, Space)
    DOM, nnodes = Shapes.DOM, Shapes.nnodes
    gs = dX.gs
    numqc = length(DOM)
    dim = size(gs, 2) - 2
    coord = gs[:, 1:dim]
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        Oloc = Array{Float64}(undef, nvec, nvec)
        row[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)
        col[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)
        @inbounds for a in 1:nvec
            aMeasure = SingleEFGMeasure(Shapes, ind, a)
            @simd for b in 1:nvec
                bMeasure = SingleEFGMeasure(Shapes, ind, b)
                Oloc[a, b], _ = f(aMeasure, bMeasure)
                Oloc[a, b] = Oloc[a, b] * gs[ind, end] * gs[ind, end-1]
            end
        end
        val[pos:pos+nvec^2-1] = vec(Oloc)
        pos += nvec^2
    end
    O = sparse(row, col, val, nnodes, nnodes)
    row = col = val = DOM = Shapes = dX = nothing
    GC.gc()
    return O
end