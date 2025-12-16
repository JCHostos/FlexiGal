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
    _, dX = f(nothing, nothing)
    Shapes = EFG_Measure(dX, Space)
    
    DOM, nnodes = Shapes.DOM, Shapes.nnodes
    dX = merge(dX)
    gs = dX.gs
    numqc = length(DOM)
    dim = size(gs, 2) - 2
    coords = gs[:, 1:dim]
    total_length = sum(x -> length(x)^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        Oloc = Array{Float64}(undef, nvec, nvec)
        row[pos:pos+nvec^2-1] = repeat(dom, outer=nvec)
        col[pos:pos+nvec^2-1] = repeat(dom, inner=nvec)
        measures = [SingleEFGMeasure(Shapes, ind, a, coords[ind, :]) for a in 1:nvec]
        weightjac = gs[ind, end] * gs[ind, end-1]
        @inbounds for a in 1:nvec
            @simd for b in 1:nvec
                Oloc[a, b], _ = f(measures[a], measures[b])
                Oloc[a, b] = Oloc[a, b] * weightjac
                #val[pos + (b-1)*nvec + (a-1)] = f(measures[a], measures[b])[1] * weightjac
            end
        end
        val[pos:pos+nvec^2-1] = Oloc[:]
        pos += nvec^2
    end
    O = sparse(row, col, val, nnodes, nnodes)
    row = col = val = DOM = Shapes = dX = nothing
    return O
end

function Linear_Assembler(f::Function, Space::EFGSpace)
    _, dX = f(nothing)
    Shapes = EFG_Measure(dX, Space)
    DOM, nnodes = Shapes.DOM, Shapes.nnodes
    dX = merge(dX)
    gs = dX.gs
    numqc = length(DOM)
    dim = size(gs, 2) - 2
    coords = gs[:, 1:dim]
    total_length = sum(length.(DOM))
    row = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    pos = 1
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        row[pos:pos+nvec-1] = DOM[ind]
        Qloc = Vector{Float64}(undef, nvec)
        weightjac=gs[ind, end] * gs[ind, end-1]
        @inbounds for a in 1:nvec
            aMeasure = SingleEFGMeasure(Shapes, ind, a, coords[ind, :])
            aux = unit_measure(aMeasure)
            Value, _ = f(aMeasure)
            Qloc[a] = aux * Value
            Qloc[a] = Qloc[a] * weightjac
        end
        val[pos:pos+nvec-1] = Qloc
        pos += nvec
    end
    Qpf = vec(Array(sparse(row, ones(Int, length(row)), val, nnodes, 1)))
    row = val = DOM = Shapes = dX = nothing
    return Qpf
end

function Linear_Problem(Bi_op::Function, Li_op::Function, Space::EFGSpace)
    # Ensamblaje volumétrico obligatorio
    A = Bilinear_Assembler(Bi_op, Space)
    F = Linear_Assembler(Li_op, Space)

    n = Space.nnodes
    α = 1e5

    if isempty(Space.Dirichlet_Measures)
        # No hay Dirichlet: Ap y Fp nulos
        Af = A
        Ff = F
    else
        # Hay Dirichlet: ensamblar penalización y contribuciones de carga
        dΓd = Space.Dirichlet_Measures
        dirichlet_values = isempty(Space.Dirichlet_Values) ?
                           fill(0.0, length(dΓd)) : Space.Dirichlet_Values
        Ap = Bilinear_Assembler((δu, u) -> ∫(δu * (α * u))dΓd, Space)

        Fp = zeros(n)
        for (diri, dΓc) in zip(dirichlet_values, dΓd)
            if diri isa Function
                # Dirichlet no homogénea dependiente de las coordenadas
                Fp += Linear_Assembler(δu -> ∫(α * (diri * δu))dΓc, Space)
            elseif diri != 0.0
                # Dirichlet constante no nula
                Fp += Linear_Assembler(δu -> ∫(α * (diri * δu))dΓc, Space)
            end
        end
        Af = A + Ap
        Ff = F + Fp
    end
    return (Af, Ff)
end


function Linear_Problem(Bi_op::Function, Space::EFGSpace)
    # Ensamblaje volumétrico obligatorio
    A = Bilinear_Assembler(Bi_op, Space)

    n = Space.nnodes
    α = 1e5

    if isempty(Space.Dirichlet_Measures)
        # No hay Dirichlet: devolvemos A y vector vacío
        return (A, zeros(n))
    else
        dΓd = Space.Dirichlet_Measures
        dirichlet_values = isempty(Space.Dirichlet_Values) ?
                           fill(0.0, length(dΓd)) : Space.Dirichlet_Values
        Ap = Bilinear_Assembler((δu, u) -> ∫(δu * (α * u))dΓd, Space)
        Ff = zeros(n)
        for (diri, dΓc) in zip(dirichlet_values, dΓd)
            if diri isa Function
                # Dirichlet no homogénea dependiente de las coordenadas
                Ff += Linear_Assembler(δu -> ∫(α * (diri * δu))dΓc, Space)
            elseif diri != 0.0
                # Dirichlet constante no nula
                Ff += Linear_Assembler(δu -> ∫(α * (diri * δu))dΓc, Space)
            end
        end
        Af = A + Ap
        return (Af, Ff)
    end
end