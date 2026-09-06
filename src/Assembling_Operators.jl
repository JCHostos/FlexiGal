using SparseArrays
function _assemble_scalar!(row, col, val, f_pure::F, Shapes::FlexiMeasure{DG}, DOM, gs, coords, numqc) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        weightjac = gs[ind, end] * gs[ind, end-1]
       # c_raw = @view coords[ind, :]
       # coord_v = SVector{DG,Float64}(c_raw)
       coord_v = SVector{DG,Float64}(ntuple(i -> coords[ind, i], Val(DG)))
       phi_point = ALL_PHI[ind]    # Vector{Float64}
        dphi_point = ALL_DPHI[ind]  # Vector{SVector{DG, Float64}}
        @inbounds for a in 1:nvec
            sa = SingleFlexiMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            for b in 1:nvec
                sb = SingleFlexiMeasure{DG}(phi_point[b], dphi_point[b], coord_v, ind)
                res_eval = f_pure(sa, sb)
                row[pos] = dom[a]
                col[pos] = dom[b]
                val[pos] = res_eval * weightjac
                pos += 1
            end
        end
    end
end

function _assemble_vector!(row, col, val, f_pure::F, Shapes::FlexiMeasure{DG}, DOM, gs, coords, numqc, D) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI

    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        nvec = length(dom)
        weightjac = gs[ind, end] * gs[ind, end-1]

        #c_raw = @view coords[ind, :]
        #coord_v = SVector{DG,Float64}(c_raw)
        coord_v = SVector{DG,Float64}(ntuple(i -> coords[ind, i], Val(DG)))
        phi_point = ALL_PHI[ind]
        dphi_point = ALL_DPHI[ind]

        @inbounds for a in 1:nvec
            # Creamos el VectorField para el nodo I (mismo phi en todas las direcciones)
            sa = SingleFlexiMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            va = if D == 2
                VectorField{2,typeof(sa)}((sa, sa))
            else
                VectorField{3,typeof(sa)}((sa, sa, sa))
            end
            node_i = dom[a]

            @inbounds for b in 1:nvec
                sb = SingleFlexiMeasure{DG}(phi_point[b], dphi_point[b], coord_v, ind)
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

function Bilinear_Assembler(f_pure::F, Space::FlexiSpace) where {F<:Function}
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    dX = Space.Measures::Vector{DomainMeasure}
    Shapes = Flexi_Measure(dX, Space)
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
function _assemble_scalar_linear!(Qpf, f_pure::F, Shapes::FlexiMeasure{DG}, DOM, gs, coords, numqc) where {F,DG}
    pos = 1
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        weightjac = gs[ind, end] * gs[ind, end-1]
        phi_point = ALL_PHI[ind]    # Vector{Float64}
        dphi_point = ALL_DPHI[ind]  # Vector{SVector{DG, Float64}}
        #c_raw = @view coords[ind, :]
        #coord_v = SVector{DG,Float64}(c_raw)
        coord_v = SVector{DG,Float64}(ntuple(i -> coords[ind, i], Val(DG)))
        @inbounds for a in 1:length(dom)
            aMeasure = SingleFlexiMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
            res_eval = f_pure(aMeasure)
            Qpf[dom[a]] += res_eval * weightjac
        end
    end
end

function _assemble_vector_linear!(Qpf, f_pure::F, Shapes::FlexiMeasure{DG}, DOM, gs, coords, numqc, D) where {F,DG}
    ALL_PHI = Shapes.PHI
    ALL_DPHI = Shapes.DPHI
    @inbounds for ind in 1:numqc
        dom = DOM[ind]
        weightjac = gs[ind, end] * gs[ind, end-1]
        #c_raw = @view coords[ind, :]
        #coord_v = SVector{DG,Float64}(c_raw)
        coord_v = SVector{DG,Float64}(ntuple(i -> coords[ind, i], Val(DG)))
        phi_point = ALL_PHI[ind]
        dphi_point = ALL_DPHI[ind]
        for a in 1:length(dom)
            sa = SingleFlexiMeasure{DG}(phi_point[a], dphi_point[a], coord_v, ind)
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

function Linear_Assembler(f_pure::F, Space::FlexiSpace) where {F<:Function}
    D = field_dim(Space.Field_Type)
    ndofs = D * Space.nnodes
    dX = Space.Measures
    Shapes = Flexi_Measure(dX, Space)
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

function get_closest_degree(model, target_degree::Int)
    nnodes_per_cell = size(model.conn, 2)
    dim = size(model.x, 2)
    available_degrees = if nnodes_per_cell == 4 && dim == 3
        [1, 3, 4]
    else
        return target_degree 
    end
    for deg in available_degrees
        if deg >= target_degree
            return deg
        end
    end 
    return maximum(available_degrees)
end

function Linear_Problem(Bi_op::Union{Integrated,MultiIntegrated}, Li_op::Union{Integrated,MultiIntegrated,Nothing}, recipe::ApproxSpace)
    res1 = Bi_op
    D = field_dim(recipe.Field_Type)
    ncells = recipe.model.ncells
    nnodes = recipe.model.nnodes
    
    # Determinamos los GDL sin mutar la malla
    is_p0 = (recipe.order == 0)
    target_nnodes = is_p0 ? ncells : nnodes
    ndofs = D * target_nnodes
    
    max_deg = 1
    Spaces = Dict{IntegrationSet,FlexiSpace}()
    
    # Función auxiliar para construir/corregir el espacio sin duplicar código
    function get_space(iset)
        if !haskey(Spaces, iset)
            sp = build_space(recipe, [iset])
            if is_p0 && sp.nnodes != ncells
                sp = FlexiSpace(sp.domain, sp.boundary, sp.Field_Type, sp.Measures, ncells)
            end
            Spaces[iset] = sp
        end
        return Spaces[iset]
    end

    A = spzeros(Float64, ndofs, ndofs)
    
    # --- Ensamblaje Bilineal ---
    if res1 isa Integrated
        f_pure = res1.arg
        max_deg = max(max_deg, res1.b.degree)
        Space = get_space(res1.b)
        A += Bilinear_Assembler(f_pure, Space)
    elseif res1 isa MultiIntegrated
        for term in res1.terms
            f_pure = term.arg
            max_deg = max(max_deg, term.b.degree)
            Space = get_space(term.b)
            A += Bilinear_Assembler(f_pure, Space)
        end
    end

    # --- Ensamblaje Lineal ---
    F = zeros(Float64, ndofs)
    if Li_op !== nothing
        res2 = Li_op
        function extract_phi(m, dim)
            if dim == 1
                return m.phi
            else
                return SVector{dim,Float64}(ntuple(i -> m[i].phi, Val(dim)))
            end
        end
        if res2 isa Integrated
            f_pure = (sa) -> extract_phi(res2.arg(sa), D)
            max_deg = max(max_deg, res2.b.degree)
            Space = get_space(res2.b)
            F = Linear_Assembler(f_pure, Space)
        elseif res2 isa MultiIntegrated
            for term in res2.terms
                f_pure = (sa) -> extract_phi(term.arg(sa), D)
                max_deg = max(max_deg, term.b.degree)
                Space = get_space(term.b)
                F .+= Linear_Assembler(f_pure, Space)
            end
        end
    end

    # --- Condiciones de Dirichlet (Tu código original intacto) ---
    Ap = spzeros(Float64, ndofs, ndofs)
    Fp = zeros(Float64, ndofs)
    
    # Protección por si ensamblamos solo matriz de masa (A = 0)
    α = isempty(A.nzval) ? 1e10 : maximum(abs, A) * 100

    if !isempty(recipe.Dirichlet_Boundaries)
        # Aseguramos que el grado de integración sea suficiente
        dsets = map(recipe.Dirichlet_Boundaries) do tri
            valid_deg = get_closest_degree(recipe.model, max_deg)
            IntegrationSet(tri, valid_deg)
        end
        
        if !haskey(Spaces, dsets[1])
            Space_D = build_space(recipe, dsets)
            for s in dsets
                Spaces[s] = Space_D
            end
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
                    if D == 1
                        return m.phi
                    end
                    return SVector{D,Float64}(ntuple(i -> m[i].phi, Val(D)))
                end
                temp_space = FlexiSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                Fp .+= Linear_Assembler(f_pure_diri, temp_space)
            end
        end
        op_arg = (D == 1) ? ((δu, u) -> α * δu * u) : ((δu, u) -> α * (δu ⋅ u))
        Ap = Bilinear_Assembler(op_arg, Space_D)
    end

    return A + Ap, F + Fp, Spaces
end

Linear_Problem(Bi_op::Union{Integrated,MultiIntegrated}, recipe::ApproxSpace) = Linear_Problem(Bi_op, nothing, recipe)

Linear_Problem(Bi_op::Function, Li_op::Union{Function,Nothing}, recipe::ApproxSpace) = error("Usa el macro @WeakForm para definir tus operadores.")

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
            f_pure = term.arg
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

function Linear_Problem(Bi_op::Union{Integrated,MultiIntegrated}, Li_op::Union{Integrated,MultiIntegrated,Nothing}, recipe::ApproxSpace, op)
    _, _, Spaces = op
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
                return SVector{dim,Float64}(ntuple(i -> m[i].phi, Val(dim)))
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
    α = maximum(A) * 100
    if !isempty(recipe.Dirichlet_Boundaries)
        dsets = map(recipe.Dirichlet_Boundaries) do tri
            valid_deg = get_closest_degree(recipe.model, max_deg)
            IntegrationSet(tri, valid_deg)
        end
        Space_D = Spaces[dsets[1]]
        dΓd = Space_D.Measures
        if !isempty(recipe.Dirichlet_Values)
            for (diri, dΓc) in zip(recipe.Dirichlet_Values, dΓd)
                f_pen = (D == 1) ? (δu -> α * diri * δu) : (δu -> δu ⋅ (diri * α))
                f_pure_diri = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1
                        return m.phi
                    end
                    return SVector{D,Float64}(ntuple(i -> m[i].phi, Val(D)))
                end
                temp_space = FlexiSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                Fp .+= Linear_Assembler(f_pure_diri, temp_space)
            end
        end
        op_arg = (D == 1) ? ((δu, u) -> α * δu * u) : ((δu, u) -> α * (δu ⋅ u))
        Ap += Bilinear_Assembler(op_arg, Space_D)
    end
    return A + Ap, F + Fp, Spaces
end

Linear_Problem(Bi_op::Union{Integrated,MultiIntegrated}, recipe::ApproxSpace, Spaces::Dict{IntegrationSet,FlexiSpace}) = Linear_Problem(Bi_op, nothing, recipe, Spaces)

function Reassemble_Vector!(Li_op::Union{Integrated,MultiIntegrated}, recipe::ApproxSpace, op)
    A, _, Spaces = op
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    max_deg = 1
    F = zeros(Float64, ndofs)
    function extract_phi(m, dim)
        if dim == 1
            return m.phi
        else
            return SVector{dim,Float64}(ntuple(i -> m[i].phi, Val(dim)))
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
    Fp = zeros(Float64, ndofs)
    α = maximum(A) * 100
    if !isempty(recipe.Dirichlet_Boundaries)
        dsets = map(recipe.Dirichlet_Boundaries) do tri
            valid_deg = get_closest_degree(recipe.model, max_deg)
            IntegrationSet(tri, valid_deg)
        end
        Space_D = Spaces[dsets[1]]
        dΓd = Space_D.Measures
        if !isempty(recipe.Dirichlet_Values)
            for (diri, dΓc) in zip(recipe.Dirichlet_Values, dΓd)
                f_pen = (D == 1) ? (δu -> α * diri * δu) : (δu -> δu ⋅ (diri * α))
                f_pure_diri = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1
                        return m.phi
                    end
                    return SVector{D,Float64}(ntuple(i -> m[i].phi, Val(D)))
                end
                temp_space = FlexiSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                Fp .+= Linear_Assembler(f_pure_diri, temp_space)
            end
        end
    end
    F_final = F + Fp
    return (A, F_final, Spaces)
end

@inline function _get_arg(s::SingleFlexiMeasure{DG}, ::Val{D}) where {DG, D}
    if D == 1
        return s
    elseif D == 2
        return VectorField{2, typeof(s)}((s, s))
    elseif D == 3
        return VectorField{3, typeof(s)}((s, s, s))
    else
        return VectorField(ntuple(_ -> s, Val(D))...)
    end
end

@inline function _extract_val(res, i, j, Di, Dj)
    if Di == 1 && Dj == 1; return res
    elseif Di > 1 && Dj == 1; return res[i]
    elseif Di == 1 && Dj > 1; return res[j]
    else; return res[i, j]; end
end

struct BlockEvaluator{F, N, I, J} f::F end
struct VectorBlockEvaluator{F, N, I} f::F end

@inline (ev::VectorBlockEvaluator{F, N, I})(mi) where {F, N, I} = 
    ev.f(ntuple(k -> k == I ? mi : nothing, Val(N)))

@inline (ev::BlockEvaluator{F, N, I, J})(mi, mj) where {F, N, I, J} = 
    ev.f(ntuple(k -> k == I ? mi : nothing, Val(N)), ntuple(k -> k == J ? mj : nothing, Val(N)))

function Bilinear_Assembler(f_pure::F, MS::MultiApproxSpace, iset::IntegrationSet, cache_func) where {F}
    N = length(MS.spaces)
    ndofs = MS.ndofs
    active_blocks = Vector{Tuple{Int, Int, FlexiSpace, FlexiSpace, FlexiMeasure, FlexiMeasure, Matrix{Float64}, Int, Int}}()
    total_length = 0
    for i in 1:N, j in 1:N
        sp_i, sp_j = cache_func(i, iset), cache_func(j, iset)
        Di, Dj = field_dim(sp_i.Field_Type), field_dim(sp_j.Field_Type)
        dummy_i = _get_arg(SingleFlexiMeasure(0.0, zero(SVector{2,Float64}), zero(SVector{2,Float64}), 1), Val(Di))
        dummy_j = _get_arg(SingleFlexiMeasure(0.0, zero(SVector{2,Float64}), zero(SVector{2,Float64}), 1), Val(Dj))
        if f_pure(ntuple(k->k==i ? dummy_i : nothing, Val(N)), ntuple(k->k==j ? dummy_j : nothing, Val(N))) !== nothing
            Sh_i = Flexi_Measure(sp_i.Measures, sp_i)
            Sh_j = Flexi_Measure(sp_j.Measures, sp_j)
            gs_block = merge(sp_i.Measures).gs 
            blk_len = sum(ind -> length(Sh_i.DOM[ind]) * length(Sh_j.DOM[ind]) * Di * Dj, 1:length(Sh_i.DOM))
            total_length += blk_len
            push!(active_blocks, (i, j, sp_i, sp_j, Sh_i, Sh_j, gs_block, MS.offsets[i], MS.offsets[j]))
        end
    end
    row = Vector{Int}(undef, total_length); col = Vector{Int}(undef, total_length); val = Vector{Float64}(undef, total_length)
    pos = 1
    for blk in active_blocks
    (i, j, sp_i, sp_j, Sh_i, Sh_j, gs_block, off_i, off_j) = blk
    Di = field_dim(sp_i.Field_Type)
    Dj = field_dim(sp_j.Field_Type)
    DG = size(gs_block, 2) - 2
    
    pos = __fill_block_indices!(row, col, val, pos, f_pure, sp_i, sp_j, Sh_i, Sh_j, gs_block, 
                                Val(i), Val(j), off_i, off_j, Val(N), 
                                Val(Di), Val(Dj), Val(DG))
    end
    if pos <= total_length
        resize!(row, pos - 1); resize!(col, pos - 1); resize!(val, pos - 1)
    end
    return sparse(row, col, val, ndofs, ndofs)
end

function __fill_block_indices!(row, col, val, pos, f_pure::F, sp_i, sp_j, Sh_i, Sh_j, gs_block, 
                               ::Val{I}, ::Val{J}, off_i, off_j, ::Val{N}, 
                               ::Val{Di}, ::Val{Dj}, ::Val{DG}) where {F, I, J, N, Di, Dj, DG}
    evaluator = BlockEvaluator{F, N, I, J}(f_pure)
    PHI_i, DPHI_i, DOM_i = Sh_i.PHI, Sh_i.DPHI, Sh_i.DOM
    PHI_j, DPHI_j, DOM_j = Sh_j.PHI, Sh_j.DPHI, Sh_j.DOM
    @inbounds for ind in 1:length(DOM_i)
        wj = gs_block[ind, end] * gs_block[ind, end-1]
        #c_raw = @view gs_block[ind, 1:DG]
        #coord_v = SVector{DG, Float64}(c_raw)
        coord_v = SVector{DG,Float64}(ntuple(i -> gs_block[ind, i], Val(DG)))
        d_i, d_j = DOM_i[ind], DOM_j[ind]
        for a in 1:length(d_i)
            arg_i = _get_arg(SingleFlexiMeasure(PHI_i[ind][a], DPHI_i[ind][a], coord_v, ind), Val(Di))
            r_base = off_i + (d_i[a] - 1) * Di
            for b in 1:length(d_j)
                arg_j = _get_arg(SingleFlexiMeasure(PHI_j[ind][b], DPHI_j[ind][b], coord_v, ind), Val(Dj))
                res = evaluator(arg_i, arg_j)
                res === nothing && continue                  
                c_base = off_j + (d_j[b] - 1) * Dj
                for jc in 1:Dj
                    curr_col = c_base + jc
                    for ir in 1:Di
                        row[pos] = r_base + ir
                        col[pos] = curr_col
                        # Usamos Di y Dj directos (son constantes aquí)
                        val[pos] = _extract_val(res, ir, jc, Di, Dj) * wj
                        pos += 1
                    end
                end
            end
        end
    end
    return pos
end

function Global_Linear_Assembler!(F_glob, f_pure::F, MS::MultiApproxSpace, iset::IntegrationSet, cache_func) where {F}
    N = length(MS.spaces)
    for i in 1:N
        sp_i = cache_func(i, iset)
        Di = field_dim(sp_i.Field_Type)
        dummy = _get_arg(SingleFlexiMeasure(0.0, zero(SVector{2,Float64}), zero(SVector{2,Float64}), 1), Val(Di))
        if f_pure(ntuple(k -> k == i ? dummy : nothing, Val(N))) !== nothing
            Shapes = Flexi_Measure(sp_i.Measures, sp_i)
            __fill_vector_core!(F_glob, f_pure, Shapes, MS.offsets[i], Di, Val(i), Val(N))
        end
    end
end

function __fill_vector_core!(F_glob, f_pure::F, Sh_i, off_i, Di, ::Val{I}, ::Val{N}) where {F, I, N}
    evaluator = VectorBlockEvaluator{F, N, I}(f_pure)
    gs = Sh_i.gs; DG = size(gs, 2) - 2
    @inbounds for ind in 1:length(Sh_i.DOM)
        wj = gs[ind, end] * gs[ind, end-1]
        #coord_v = SVector{DG, Float64}(@view gs[ind, 1:DG])
        coord_v = SVector{DG,Float64}(ntuple(i -> gs[ind, i], Val(DG)))
        dom_i = Sh_i.DOM[ind]
        for a in 1:length(dom_i)
            arg_i = _get_arg(SingleFlexiMeasure(Sh_i.PHI[ind][a], Sh_i.DPHI[ind][a], coord_v, ind), Val(Di))
            res = evaluator(arg_i)
            res === nothing && continue
            for ir in 1:Di
                val = (Di == 1) ? res.phi : res[ir].phi
                F_glob[off_i + (dom_i[a] - 1) * Di + ir] += val * wj
            end
        end
    end
end

function Linear_Problem(Bi_op::Union{Integrated,MultiIntegrated}, Li_op::Union{Integrated,MultiIntegrated,Nothing}, MS::MultiApproxSpace)
    N = length(MS.spaces)
    ndofs = MS.ndofs
    max_deg = 1
    Spaces_Cache = Dict{Tuple{Int, IntegrationSet}, FlexiSpace}()
    function get_cached_space(idx, iset)
        return get!(Spaces_Cache, (idx, iset)) do
            build_space(MS.spaces[idx], [iset])
        end
    end
    A = spzeros(Float64, ndofs, ndofs)
    F = zeros(Float64, ndofs)
    bi_terms = (Bi_op isa Integrated) ? [Bi_op] : Bi_op.terms
    for term in bi_terms
        iset = term.b
        max_deg = max(max_deg, iset.degree)
        A += Bilinear_Assembler(term.arg, MS, iset, get_cached_space)
    end
    if Li_op !== nothing
        li_terms = (Li_op isa Integrated) ? [Li_op] : Li_op.terms
        for term in li_terms
            iset = term.b
            max_deg = max(max_deg, iset.degree)
            Global_Linear_Assembler!(F, term.arg, MS, iset, get_cached_space)
        end
    end
    α = isempty(A.nzval) ? 1e10 : maximum(abs, A) * 100
    for i in 1:N
        recipe = MS.spaces[i]
        off = MS.offsets[i]
        if recipe.constraint == :ZeroMean
            A[off + 1, off + 1] += α
        end
        if !isempty(recipe.Dirichlet_Boundaries)
            off = MS.offsets[i]
            D = field_dim(recipe.Field_Type)
            dsets = map(recipe.Dirichlet_Boundaries) do tri
            valid_deg = get_closest_degree(recipe.model, max_deg)
            IntegrationSet(tri, valid_deg)
            end
            Space_D = build_space(recipe, dsets)
            op_arg = (D == 1) ? ((δu, u) -> α * δu * u) : ((δu, u) -> α * (δu ⋅ u))
            Ap_local = Bilinear_Assembler(op_arg, Space_D) 
            I, J, V = findnz(Ap_local)
            A += sparse(I .+ off, J .+ off, V, ndofs, ndofs)
            if !isempty(recipe.Dirichlet_Values)
                for (diri, dΓc) in zip(recipe.Dirichlet_Values, Space_D.Measures)
                    f_pen = (D == 1) ? (v -> α * diri * v) : (v -> v ⋅ (diri * α))
                    f_pure_diri = (v) -> begin
                        m = f_pen(v)
                        return (D == 1) ? m.phi : SVector{D,Float64}(ntuple(k -> m[k].phi, Val(D)))
                    end             
                    temp_sp = FlexiSpace(Space_D.domain, Space_D.boundary, recipe.Field_Type, [dΓc], Space_D.nnodes)
                    F_local = Linear_Assembler(f_pure_diri, temp_sp)
                    F[off+1 : off+length(F_local)] .+= F_local
                end
            end
        end
    end
    return A, F, Spaces_Cache, MS
end