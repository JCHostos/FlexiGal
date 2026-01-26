using SparseArrays

function Bilinear_Assembler(f::Function, Space::EFGSpace)
    res = f(nothing, nothing)
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    if res isa Integrated
        dX = res.b
        Shapes = EFG_Measure(dX, Space)
        DOM, nnodes = Shapes.DOM, Shapes.nnodes
        dX = merge(dX)
        gs = dX.gs
        numqc = length(DOM)
        dim_geo = size(gs, 2) - 2
        coords = gs[:, 1:dim_geo]
        total_length = sum(x -> (length(x)^2) * D^2, DOM)
        row = Vector{Int}(undef, total_length)
        col = Vector{Int}(undef, total_length)
        val = Vector{Float64}(undef, total_length)
        pos = 1

        if D == 1
            #--- SCALAR FIELD---#
            @inbounds for ind in 1:numqc
                dom = DOM[ind]
                nvec = length(dom)
                weightjac = gs[ind, end] * gs[ind, end-1]
                coord=coords[ind,:]
                @inbounds for a in 1:nvec
                    I = dom[a]
                    scalar_a = SingleEFGMeasure(Shapes, ind, a, coord)
                    @inbounds for b in 1:nvec
                        J = dom[b]
                        scalar_b = SingleEFGMeasure(Shapes, ind, b, coord)
                        res_eval = f(scalar_a, scalar_b)
                        Kab = res_eval.arg
                        row[pos] = I
                        col[pos] = J
                        val[pos] = Kab * weightjac
                        pos += 1
                    end
                end
            end # for ind
        else
            # --- VECTOR FIELD ---#
            @inbounds for ind in 1:numqc
                dom = DOM[ind]
                nvec = length(dom)
                weightjac = gs[ind, end] * gs[ind, end-1]
                coord=coords[ind,:]
                @inbounds for a in 1:nvec
                    I = dom[a]
                    offset_i = (I - 1) * D  # Pre-calculamos el offset de fila
                    scalar_a = SingleEFGMeasure(Shapes, ind, a, coord)
                    v_meas_a = VectorField(ntuple(i -> scalar_a, D)...)
                    @inbounds for b in 1:nvec
                        J = dom[b]
                        offset_j = (J - 1) * D # Pre-calculamos el offset de columna
                        scalar_b = SingleEFGMeasure(Shapes, ind, b, coord)
                        v_meas_b = VectorField(ntuple(i -> scalar_b, D)...)
                        res_eval = f(v_meas_a, v_meas_b)
                        Kab = res_eval.arg # Matriz DxD

                        # --- Ensamblaje del bloque DxD optimizado ---
                        for i in 1:D
                            r_idx = offset_i + i
                            @inbounds @simd for j in 1:D
                                c_idx = offset_j + j
                                row[pos] = r_idx
                                col[pos] = c_idx
                                val[pos] = Kab[i, j] * weightjac
                                pos += 1
                            end
                        end
                    end
                end
            end
        end # if D == 1

        # 5. Construcción y limpieza (común a ambos casos)
        O = sparse(row, col, val, ndofs, ndofs)
        row = col = val = DOM = Shapes = dX = nothing

        return O
    elseif res isa MultiIntegrated
        K_total = spzeros(Float64, ndofs, ndofs)
        for ii in 1:length(res.terms)
            dX = res.terms[ii].b
            Shapes = EFG_Measure(dX, Space)
            DOM, nnodes = Shapes.DOM, Shapes.nnodes
            dX = merge(dX)
            gs = dX.gs
            numqc = length(DOM)
            dim_geo = size(gs, 2) - 2
            coords = gs[:, 1:dim_geo]
            total_length = sum(x -> (length(x)^2) * D^2, DOM)
            row = Vector{Int}(undef, total_length)
            col = Vector{Int}(undef, total_length)
            val = Vector{Float64}(undef, total_length)
            pos = 1

            if D == 1
                #--- SCALAR FIELD---#
                @inbounds for ind in 1:numqc
                    dom = DOM[ind]
                    nvec = length(dom)
                    weightjac = gs[ind, end] * gs[ind, end-1]
                    coord=coords[ind,:]
                    @inbounds for a in 1:nvec
                        I = dom[a]
                        scalar_a = SingleEFGMeasure(Shapes, ind, a, coord)
                        @inbounds for b in 1:nvec
                            J = dom[b]
                            scalar_b = SingleEFGMeasure(Shapes, ind, b, coord)
                            res_eval = f(scalar_a, scalar_b).terms[ii]
                            Kab = res_eval.arg
                            row[pos] = I
                            col[pos] = J
                            val[pos] = Kab * weightjac
                            pos += 1
                        end
                    end
                end # for ind
            else
                # --- VECTOR FIELD ---#
                @inbounds for ind in 1:numqc
                    dom = DOM[ind]
                    nvec = length(dom)
                    weightjac = gs[ind, end] * gs[ind, end-1]
                    coord=coords[ind,:]
                    @inbounds for a in 1:nvec
                        I = dom[a]
                        scalar_a = SingleEFGMeasure(Shapes, ind, a, coord)
                        v_meas_a = VectorField(ntuple(i -> scalar_a, D)...)
                        @inbounds for b in 1:nvec
                            J = dom[b]
                            scalar_b = SingleEFGMeasure(Shapes, ind, b, coord)
                            v_meas_b = VectorField(ntuple(i -> scalar_b, D)...)
                            res_eval = f(v_meas_a, v_meas_b).terms[ii]
                            Kab = res_eval.arg

                            # Ensamblaje del bloque DxD
                            @inbounds for i in 1:D
                                r_idx = (I - 1) * D + i
                                @inbounds for j in 1:D
                                    c_idx = (J - 1) * D + j
                                    row[pos] = r_idx
                                    col[pos] = c_idx
                                    val[pos] = Kab[i, j] * weightjac
                                    pos += 1
                                end
                            end
                        end
                    end
                end # for ind
            end # if D == 1

            # 5. Construcción y limpieza (común a ambos casos)
            K_total += sparse(row, col, val, ndofs, ndofs)
            row = col = val = DOM = Shapes = dX = nothing
        end
        return K_total
    end
end

function Linear_Assembler(f::Function, Space::EFGSpace)
    res_init = f(nothing) # Evaluamos primero para ver qué es
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    if res_init isa Integrated
        # --- CASO INTEGRAL ÚNICA ---
        dX = res_init.b
        Shapes = EFG_Measure(dX, Space)
        DOM = Shapes.DOM
        dX_m = merge(dX)
        gs = dX_m.gs
        numqc = length(DOM)
        dim_geo = size(gs, 2) - 2
        coords = gs[:, 1:dim_geo]

        total_length = sum(length.(DOM)) * D
        row = Vector{Int}(undef, total_length)
        val = Vector{Float64}(undef, total_length)
        pos = 1

        if D == 1
            @inbounds for ind in 1:numqc
                dom = DOM[ind]
                weightjac = gs[ind, end] * gs[ind, end-1]
                @inbounds for a in 1:length(dom)
                    aMeasure = SingleEFGMeasure(Shapes, ind, a, coords[ind, :])
                    # No pisamos 'res_init', usamos una variable local
                    res_eval = f(aMeasure)
                    Valuei = res_eval.arg.phi
                    row[pos] = dom[a]
                    val[pos] = Valuei * weightjac
                    pos += 1
                end
            end
        else
            @inbounds for ind in 1:numqc
                dom = DOM[ind]
                weightjac = gs[ind, end] * gs[ind, end-1]
                scalar_measures = [SingleEFGMeasure(Shapes, ind, a, coords[ind, :]) for a in 1:length(dom)]
                v_measures = [VectorField(ntuple(i -> scalar_measures[a], D)...) for a in 1:length(dom)]
                @inbounds for a in 1:length(dom)
                    I = dom[a]
                    res_eval = f(v_measures[a])
                    Fa = res_eval.arg
                    for i in 1:D
                        row[pos] = (I - 1) * D + i
                        val[pos] = Fa[i].phi * weightjac
                        pos += 1
                    end
                end
            end
        end
        Qpf = vec(Array(sparse(row, ones(Int, length(row)), val, ndofs, 1)))
        return Qpf

    elseif res_init isa MultiIntegrated
        # --- CASO SUMA DE INTEGRALES ---
        F_total = zeros(Float64, ndofs)

        for ii in 1:length(res_init.terms)
            # Extraemos el dominio del término ii
            dX_term = res_init.terms[ii].b
            Shapes = EFG_Measure(dX_term, Space)
            DOM, _ = Shapes.DOM, Shapes.nnodes
            dX_m = merge(dX_term)
            gs = dX_m.gs
            numqc = length(DOM)
            dim_geo = size(gs, 2) - 2
            coords = gs[:, 1:dim_geo]

            total_length = sum(length.(DOM)) * D
            row = Vector{Int}(undef, total_length)
            val = Vector{Float64}(undef, total_length)
            pos = 1

            if D == 1
                @inbounds for ind in 1:numqc
                    dom = DOM[ind]
                    weightjac = gs[ind, end] * gs[ind, end-1]
                    @inbounds for a in 1:length(dom)
                        aMeasure = SingleEFGMeasure(Shapes, ind, a, coords[ind, :])
                        # Extraemos solo el término ii de la evaluación
                        res_eval = f(aMeasure).terms[ii]
                        Valuei = res_eval.arg.phi
                        row[pos] = dom[a]
                        val[pos] = Valuei * weightjac
                        pos += 1
                    end
                end
            else
                @inbounds for ind in 1:numqc
                    dom = DOM[ind]
                    weightjac = gs[ind, end] * gs[ind, end-1]
                    scalar_measures = [SingleEFGMeasure(Shapes, ind, a, coords[ind, :]) for a in 1:length(dom)]
                    v_measures = [VectorField(ntuple(i -> scalar_measures[a], D)...) for a in 1:length(dom)]
                    @inbounds for a in 1:length(dom)
                        I = dom[a]
                        offset = (I - 1) * D
                        res_eval = f(v_measures[a]).terms[ii]
                        Fa = res_eval.arg
                        @inbounds @simd for i in 1:D
                            row[pos] = offset + i
                            val[pos] = Fa[i].phi * weightjac
                            pos += 1
                        end
                    end
                end
            end
            # Sumamos este aporte al vector total
            F_total .+= vec(Array(sparse(row, ones(Int, length(row)), val, ndofs, 1)))
            row = val = DOM = Shapes = dX_term = dX_m = nothing
        end
        return F_total
    end
end


function Linear_Problem(Bi_op::Function, Li_op::Union{Function,Nothing}, Space::EFGSpace)
    A = Bilinear_Assembler(Bi_op, Space)

    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    α = 1e6

    F = isnothing(Li_op) ? zeros(ndofs) : Linear_Assembler(Li_op, Space)

    if isempty(Space.Dirichlet_Measures)
        return (A, F)
    else
        dΓd = Space.Dirichlet_Measures
        # Si no hay valores, llenamos con "ceros" del tipo correcto
        dirichlet_values = isempty(Space.Dirichlet_Values) ?
                           fill(D == 1 ? 0.0 : VectorField(ntuple(i -> 0.0, D)...), length(dΓd)) :
                           Space.Dirichlet_Values

        # --- PENALIZACIÓN DE MATRIZ ---
        op_mat = D == 1 ? (δu, u) -> ∫(δu * (α * u))dΓd : (δu, u) -> ∫(α * (δu ⋅ u))dΓd
        Ap = Bilinear_Assembler(op_mat, Space)

        # --- PENALIZACIÓN DE VECTOR ---
        Fp = zeros(ndofs)
        for (diri, dΓc) in zip(dirichlet_values, dΓd)

            f_pen = if D == 1
                # Caso Escalar (diri es Número o Función -> Número)
                δu -> ∫(α * (diri * δu))dΓc
            else
                # Caso Vectorial (diri es VectorField o Función -> VectorField)
                # Tu Linear_Assembler vectorial ya sabe procesar (δu ⋅ VectorField)
                δu -> ∫((δu ⋅ (diri * α)))dΓc
            end

            Fp += Linear_Assembler(f_pen, Space)
        end

        return (A + Ap, F + Fp)
    end
end

Linear_Problem(Bi_op::Function, Space::EFGSpace) = Linear_Problem(Bi_op, nothing, Space)