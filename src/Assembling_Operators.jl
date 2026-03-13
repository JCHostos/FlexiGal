using SparseArrays
function _assemble_scalar!(row, col, val, f_pure::F, Shapes::EFGMeasure{DG}, DOM, gs, coords, numqc) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        weightjac = gs[ind, end] * gs[ind, end-1]
        c_raw = @view coords[ind, :]
        coord_v = SVector{DG,Float64}(c_raw)
        phi_point = ALL_PHI[ind]    # Vector{Float64}
        dphi_point = ALL_DPHI[ind]  # Vector{SVector{DG, Float64}}
        @inbounds for a in 1:nvec
            sa = SingleEFGMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            for b in 1:nvec
                sb = SingleEFGMeasure{DG}(phi_point[b], dphi_point[b], coord_v, ind)
                res_eval = f_pure(sa, sb)
                row[pos] = dom[a]
                col[pos] = dom[b]
                val[pos] = res_eval * weightjac
                pos += 1
            end
        end
    end
end

function _assemble_vector!(row, col, val, f_pure::F, Shapes::EFGMeasure{DG}, DOM, gs, coords, numqc, D) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI

    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        weightjac = gs[ind, end] * gs[ind, end-1]

        c_raw = @view coords[ind, :]
        coord_v = SVector{DG,Float64}(c_raw)

        phi_point = ALL_PHI[ind]
        dphi_point = ALL_DPHI[ind]

        @inbounds for a in 1:nvec
            # Creamos el VectorField para el nodo I (mismo phi en todas las direcciones)
            sa = SingleEFGMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            va = if D == 2
                VectorField{2,typeof(sa)}((sa, sa))
            else
                VectorField{3,typeof(sa)}((sa, sa, sa))
            end
            node_i = dom[a]

            @inbounds for b in 1:nvec
                sb = SingleEFGMeasure{DG}(phi_point[b], dphi_point[b], coord_v, ind)
                vb = if D == 2
                    VectorField{2,typeof(sb)}((sb, sb))
                else
                    VectorField{3,typeof(sb)}((sb, sb, sb))
                end
                node_j = dom[b]
                Kab = f_pure(va, vb) * weightjac

                # Ensamblaje manual del bloque DxD (evitamos repeat y slicing)
                for j_col in 1:D
                    global_col = (node_j - 1) * D + j_col
                    for i_row in 1:D
                        global_row = (node_i - 1) * D + i_row

                        @inbounds row[pos] = global_row
                        @inbounds col[pos] = global_col
                        @inbounds val[pos] = Kab[i_row, j_col]
                        pos += 1
                    end
                end
            end
        end
    end
end

function Bilinear_Assembler(f_pure::F, Space::EFGSpace) where {F<:Function}
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    dX = Space.Measures
    Shapes = EFG_Measure(dX, Space)
    DOM = Shapes.DOM
    dX = merge(dX)
    gs = dX.gs
    numqc = length(DOM)
    dim_geo = size(gs, 2) - 2
    coords = gs[:, 1:dim_geo]
    total_length = sum(x -> (length(x)^2) * D^2, DOM)
    row = Vector{Int}(undef, total_length)
    col = Vector{Int}(undef, total_length)
    val = Vector{Float64}(undef, total_length)
    if D == 1
        _assemble_scalar!(row, col, val, f_pure, Shapes, DOM, gs, coords, numqc)
    else
        _assemble_vector!(row, col, val, f_pure, Shapes, DOM, gs, coords, numqc, D)
    end # if D == 1
    O = sparse(row, col, val, ndofs, ndofs)
    row = col = val = DOM = Shapes = dX = nothing
    return O
end
function _assemble_scalar_linear!(Qpf, f_pure::F, Shapes::EFGMeasure{DG}, DOM, gs, coords, numqc) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        weightjac = gs[ind, end] * gs[ind, end-1]
        phi_point = ALL_PHI[ind]    # Vector{Float64}
        dphi_point = ALL_DPHI[ind]  # Vector{SVector{DG, Float64}}
        c_raw = @view coords[ind, :]
        coord_v = SVector{DG,Float64}(c_raw)
        @inbounds for a in 1:length(dom)
            aMeasure = SingleEFGMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            res_eval = f_pure(aMeasure)
            Qpf[dom[a]] += res_eval * weightjac
        end
    end
end

function _assemble_vector_linear!(Qpf, f_pure::F, Shapes::EFGMeasure{DG}, DOM, gs, coords, numqc, D) where {F,DG}
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        weightjac = gs[ind, end] * gs[ind, end-1]
        c_raw = @view coords[ind, :]
        coord_v = SVector{DG,Float64}(c_raw)
        phi_point = ALL_PHI[ind]
        dphi_point = ALL_DPHI[ind]
        for a in 1:length(dom)
            sa = SingleEFGMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            va = VectorField(ntuple(i -> sa, Val(D))...)
            res_eval = f_pure(va)
            node_i = dom[a]
            for i in 1:D
                idx = (node_i - 1) * D + i
                Qpf[idx] += res_eval[i] * weightjac
            end
        end
    end
end

function Linear_Assembler(f_pure::F, Space::EFGSpace) where {F<:Function}
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    dX = Space.Measures
    Shapes = EFG_Measure(dX, Space)
    DOM = Shapes.DOM
    dX_m = merge(dX)
    gs = dX_m.gs
    numqc = length(DOM)
    dim_geo = size(gs, 2) - 2
    coords = gs[:, 1:dim_geo]
    Qpf = zeros(Float64, ndofs)
    if D == 1
        _assemble_scalar_linear!(Qpf, f_pure, Shapes, DOM, gs, coords, numqc)
    else
        _assemble_vector_linear!(Qpf, f_pure, Shapes, DOM, gs, coords, numqc, D)
    end
    return Qpf
end

function Linear_Problem(Bi_op::Union{Integrated, MultiIntegrated}, Li_op::Union{Integrated, MultiIntegrated, Nothing}, recipe::ApproxSpace)
    res1 = Bi_op
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    max_deg = 1
    Spaces = Dict{IntegrationSet,EFGSpace}()
    A = spzeros(Float64, ndofs, ndofs)
    if res1 isa Integrated
        f_pure = res1.arg
        isets = [res1.b]
        max_deg = max(max_deg, res1.b.degree)
        if !haskey(Spaces, res1.b)
            Space = build_space(recipe, isets)
            for s in isets; Spaces[s] = Space; end
        else
            Space = Spaces[res1.b]
        end
        A += Bilinear_Assembler(f_pure, Space)
    elseif res1 isa MultiIntegrated
        for term in res1.terms
            f_pure = term.arg
            isets = [term.b]
            max_deg = max(max_deg, term.b.degree)
            if !haskey(Spaces, term.b)
                Space = build_space(recipe, isets)
                for s in isets; Spaces[s] = Space; end
            else
                Space = Spaces[term.b]
            end
            A += Bilinear_Assembler(f_pure, Space)
        end
    end
    F = zeros(Float64, ndofs)
    if Li_op !== nothing
        res2 = Li_op
        function extract_phi(m, dim)
            if dim == 1
                return m.phi
            else
                return SVector{dim, Float64}(ntuple(i -> m[i].phi, Val(dim)))
            end
        end
        if res2 isa Integrated
            f_pure = (sa) -> extract_phi(res2.arg(sa), D)
            isets = [res2.b]
            max_deg = max(max_deg, res2.b.degree)
            if !haskey(Spaces, res2.b)
                Space = build_space(recipe, isets)
                for s in isets; Spaces[s] = Space; end
            else
                Space = Spaces[res2.b]
            end
            F = Linear_Assembler(f_pure, Space)     
        elseif res2 isa MultiIntegrated
            for term in res2.terms
                f_pure = (sa) -> extract_phi(term.arg(sa), D)
                isets = [term.b]
                max_deg = max(max_deg, term.b.degree)
                if !haskey(Spaces, term.b)
                    Space = build_space(recipe, isets)
                    for s in isets; Spaces[s] = Space; end
                else
                    Space = Spaces[term.b]
                end
                F .+= Linear_Assembler(f_pure, Space)
            end
        end
    end
    Ap = spzeros(Float64, ndofs, ndofs)
    Fp = zeros(Float64, ndofs)
    α = 1e6
    if !isempty(recipe.Dirichlet_Boundaries)
        # Aseguramos que el grado de integración sea suficiente
        dsets = [IntegrationSet(tri, max_deg) for tri in recipe.Dirichlet_Boundaries]
        if !haskey(Spaces, dsets[1])
            Space_D = build_space(recipe, dsets)
            for s in dsets; Spaces[s] = Space_D; end
        else
            Space_D = Spaces[dsets[1]]
        end        
        dΓd = Space_D.Measures
        if !isempty(recipe.Dirichlet_Values)
            for (diri, dΓc) in zip(recipe.Dirichlet_Values, dΓd)
                # Creamos el integrando de penalización
                f_pen = (D == 1) ? (δu -> α * diri * δu) : (δu -> δu ⋅ (diri * α))
                f_pure_diri = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1; return m.phi; end
                    return SVector{D, Float64}(ntuple(i -> m[i].phi, Val(D)))
                end
                temp_space = EFGSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                Fp .+= Linear_Assembler(f_pure_diri, temp_space)
            end
        end
        op_arg = (D == 1) ? ((δu, u) -> α * δu * u) : ((δu, u) -> α * (δu ⋅ u))
        Ap = Bilinear_Assembler(op_arg, Space_D)
    end

    return A + Ap, F + Fp, Spaces
end

Linear_Problem(Bi_op::Union{Integrated, MultiIntegrated}, recipe::ApproxSpace) = Linear_Problem(Bi_op, nothing, recipe)

Linear_Problem(Bi_op::Function, Li_op::Union{Function, Nothing}, recipe::ApproxSpace) = error("Usa el macro @WeakForm para definir tus operadores.")

function Sub_LinearProblem(Jacobian, Residual, recipe::ApproxSpace, Spaces)
    res_jac = Jacobian
    D = field_dim(recipe.Field_Type)
    ndofs = D * recipe.model.nnodes
    K = spzeros(Float64, ndofs, ndofs)
    R = zeros(Float64, ndofs)
    if res_jac isa Integrated
        iset = res_jac.b
        f_pure = res_jac.arg
        Space = Get_space_from_IntegrationSet(Spaces, iset)
        K .+= Bilinear_Assembler(f_pure, Space)
    elseif res_jac isa MultiIntegrated
        for ii in 1:length(res_jac.terms)
            term = res_jac.terms[ii]
            iset = term.b
            f_pure =term.arg
            Space = Get_space_from_IntegrationSet(Spaces, iset)
            K .+= Bilinear_Assembler(f_pure, Space)
        end
    end
    res_res = Residual
    if res_res isa Integrated
        iset = res_res.b
        f_pure = (sa) -> begin
                m = res_res.arg(sa)
                if D == 1
                    return m.phi
                elseif D == 2
                    return SVector{2,Float64}(m[1].phi, m[2].phi)
                elseif D == 3
                    return SVector{3,Float64}(m[1].phi, m[2].phi, m[3].phi)
                end
            end
        Space = Get_space_from_IntegrationSet(Spaces, iset)
        R .+= Linear_Assembler(f_pure, Space)
    elseif res_res isa MultiIntegrated
        for ii in 1:length(res_res.terms)
            term = res_res.terms[ii]
            iset = term.b
            Space = Get_space_from_IntegrationSet(Spaces, iset)
            f_pure = (sa) -> begin
                m = term.arg(sa)
                if D == 1
                    return m.phi
                elseif D == 2
                    return SVector{2,Float64}(m[1].phi, m[2].phi)
                elseif D == 3
                    return SVector{3,Float64}(m[1].phi, m[2].phi, m[3].phi)
                end
            end
            R .+= Linear_Assembler(f_pure, Space)
        end
    end
    return K, R
end

function Linear_Problem(Bi_op::Union{Integrated, MultiIntegrated}, Li_op::Union{Integrated, MultiIntegrated, Nothing}, recipe::ApproxSpace, Spaces::Dict{IntegrationSet,EFGSpace})
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    max_deg = 1
    A = spzeros(Float64, ndofs, ndofs)
    if Bi_op isa Integrated
        f_pure = Bi_op.arg
        iset = Bi_op.b
        max_deg = max(max_deg, iset.degree)
        Space = Spaces[iset]
        A += Bilinear_Assembler(f_pure, Space)
    elseif Bi_op isa MultiIntegrated
        for term in Bi_op.terms
            f_pure = term.arg
            iset = term.b
            max_deg = max(max_deg, iset.degree)
            Space = Spaces[iset]
            A += Bilinear_Assembler(f_pure, Space)
        end
    end
    F = zeros(Float64, ndofs)
    if Li_op !== nothing
        function extract_phi(m, dim)
            if dim == 1
                return m.phi
            else
                return SVector{dim, Float64}(ntuple(i -> m[i].phi, Val(dim)))
            end
        end
        if Li_op isa Integrated
            f_pure = (sa) -> extract_phi(Li_op.arg(sa), D)
            iset = Li_op.b
            max_deg = max(max_deg, iset.degree)
            Space = Spaces[iset]
            F .+= Linear_Assembler(f_pure, Space)     
        elseif Li_op isa MultiIntegrated
            for term in Li_op.terms
                f_pure = (sa) -> extract_phi(term.arg(sa), D)
                iset = term.b
                max_deg = max(max_deg, iset.degree)
                Space = Spaces[iset]
                F .+= Linear_Assembler(f_pure, Space)
            end
        end
    end
    Ap = spzeros(Float64, ndofs, ndofs)
    Fp = zeros(Float64, ndofs)
    α = 1e6
    if !isempty(recipe.Dirichlet_Boundaries)
        dsets = [IntegrationSet(tri, max_deg) for tri in recipe.Dirichlet_Boundaries]
        Space_D = Spaces[dsets[1]] 
        dΓd = Space_D.Measures
        if !isempty(recipe.Dirichlet_Values)
            for (diri, dΓc) in zip(recipe.Dirichlet_Values, dΓd)
                f_pen = (D == 1) ? (δu -> α * diri * δu) : (δu -> δu ⋅ (diri * α))
                f_pure_diri = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1; return m.phi; end
                    return SVector{D, Float64}(ntuple(i -> m[i].phi, Val(D)))
                end
                temp_space = EFGSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                Fp .+= Linear_Assembler(f_pure_diri, temp_space)
            end
        end
        op_arg = (D == 1) ? ((δu, u) -> α * δu * u) : ((δu, u) -> α * (δu ⋅ u))
        Ap += Bilinear_Assembler(op_arg, Space_D)
    end
    return A + Ap, F + Fp, Spaces
end

Linear_Problem(Bi_op::Union{Integrated, MultiIntegrated}, recipe::ApproxSpace, Spaces::Dict{IntegrationSet,EFGSpace}) = Linear_Problem(Bi_op, nothing, recipe, Spaces)