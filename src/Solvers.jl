function Get_space_from_IntegrationSet(Spaces::Dict{Union{IntegrationSet,PointSet},FlexiSpace}, target_iset::Union{IntegrationSet,PointSet})
    return get(Spaces, target_iset) do
        error("No se encontró espacio para el set con tag: '$(target_iset.tri.tag)'")
    end
end

Get_Measures(a::FlexiSpace) = a.Measures

function extract_FlexiFunction(field)
    if field isa FlexiFunction
        return field
    end
    if !isprimitivetype(typeof(field))
        for name in fieldnames(typeof(field))
            child = getproperty(field, name)
            result = extract_FlexiFunction(child)
            if result isa FlexiFunction
                return result
            end
        end
    end
    return nothing
end

function get_field_coordinates(f, model=nothing)
    fun = extract_FlexiFunction(f)
    if fun !== nothing
        gs = fun.Measure.gs
        d_geo = size(gs, 2) - 2
        return gs[:, 1:d_geo]
    end
    if hasfield(typeof(f), :targets)
        return f.targets
    elseif hasfield(typeof(f), :source) && hasfield(typeof(f.source), :targets)
        return f.source.targets
    elseif !isnothing(model) && hasproperty(model, :x)
        return model.x
    else
        error("No se pudieron extraer las coordenadas del campo $(typeof(f))")
    end
end

function Solve(op, iset::IntegrationSet)
    A, F, Spaces = op
    u_nodal = A \ F
    Space = Get_space_from_IntegrationSet(Spaces, iset)
    M = Space.Measures isa AbstractVector ? Space.Measures[1] : Space.Measures
    SolvedFlexifunction = FlexiFunction(u_nodal, Space, M)
    return SolvedFlexifunction
end

function Solve(op)
    A, F, Spaces = op
    u_nodal = A \ F
    Flexifunctions = Dict(s => FlexiFunction(u_nodal, sp, sp.Measures[1]) for (s, sp) in Spaces)
    return Flexifunctions
end

function Solve(op::Tuple{SparseMatrixCSC,Vector{Float64},Dict,MultiApproxSpace}, iset::IntegrationSet)
    A, F, Cache, MS = op
    u_global = A \ F
    field_results = []
    for i in 1:length(MS.spaces)
        recipe = MS.spaces[i]
        off = MS.offsets[i]
        D = field_dim(recipe.Field_Type)
        ndofs_field = (recipe.order == 0) ? (recipe.model.ncells * D) : (recipe.model.nnodes * D)
        u_field = u_global[off+1:off+ndofs_field]
        sp = Cache[(i, iset)]
        ph = FlexiFunction(u_field, sp, sp.Measures[1])
        if recipe.constraint == :ZeroMean
            if D == 1
                int_p = ∫(ph)iset
                m = sp.Measures[1]
                vol = sum(@view(m.gs[:, end-1]) .* @view(m.gs[:, end]))
                p_avg = int_p / vol
                u_field .-= p_avg
                ph = FlexiFunction(u_field, sp, sp.Measures[1])
            else
                @warn "Constraint :ZeroMean ignored for vector field (D=$D). Only applied to scalar fields."
            end
        end
        push!(field_results, ph)
    end
    return length(field_results) == 1 ? field_results[1] : tuple(field_results...)
end

struct NonLinearOperator
    step_builder::Function
    space::ApproxSpace
end

function nothing_trace(obj)
    isnothing(obj) && return true
    T = typeof(obj)
    if isprimitivetype(T) || T <: Number || T <: AbstractArray || T == String
        return false
    end
    for name in fieldnames(T)
        try
            val = getfield(obj, name)
            nothing_trace(val) && return true
        catch
        end
    end
    return false
end

function NL_Solver(NL_Op; u_seed=nothing, tol=1e-6, max_iter=15)
    recipe = NL_Op.space
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    u_nodal = (u_seed === nothing) ? zeros(Float64, ndofs) : copy(u_seed)
    max_deg = 1
    m = methods(NL_Op.step_builder)
    n_args = length(Base.method_argnames(last(collect(m)))) - 1
    arg_isets = Vector{Union{IntegrationSet,PointSet}}(undef, n_args)
    for i in 1:n_args
        test_args = Any[0.0 for _ in 1:n_args]
        test_args[i] = nothing
        Jac0, Res0 = NL_Op.step_builder(test_args...)
        encontrado = false
        for forma in [Jac0, Res0]
            terms = forma isa MultiIntegrated ? forma.terms : [forma]
            for t in terms
                if nothing_trace(t.arg)
                    arg_isets[i] = t.b
                    encontrado = true
                    break
                end
            end
            encontrado && break
        end
        !encontrado && error("No se detectó IntegrationSet para el argumento $i")
    end
    Spaces = Dict{Union{IntegrationSet,PointSet},FlexiSpace}()
    function get_space(iset)
        if !haskey(Spaces, iset)
            sp = iset isa PointSet ? build_point_space(recipe, iset) : build_space(recipe, [iset])
            Spaces[iset] = sp
        end
        return Spaces[iset]
    end

    # 2. Construcción de espacios de los argumentos
    isets_vec = unique(arg_isets)
    for s in isets_vec
        get_space(s)
        # Protegemos el grado polinomial en caso de que sea un PointSet
        if hasproperty(s, :degree)
            max_deg = max(max_deg, s.degree)
        end
    end
    Flexifunctions = Dict(s => FlexiFunction(u_nodal, sp, sp.Measures[1]) for (s, sp) in Spaces)
    uhs = [Flexifunctions[s] for s in isets_vec]
    Jac, Res = NL_Op.step_builder(uhs...)
    if Jac isa Integrated
        if hasproperty(Jac.b, :degree)
            max_deg = max(max_deg, Jac.b.degree)
        end
        get_space(Jac.b)
    elseif Jac isa MultiIntegrated
        for term in Jac.terms
            if hasproperty(term.b, :degree)
                max_deg = max(max_deg, term.b.degree)
            end
            get_space(term.b)
        end
    end

    if Res isa Integrated
        if hasproperty(Res.b, :degree)
            max_deg = max(max_deg, Res.b.degree)
        end
        get_space(Res.b)
    elseif Res isa MultiIntegrated
        for term in Res.terms
            if hasproperty(term.b, :degree)
                max_deg = max(max_deg, term.b.degree)
            end
            get_space(term.b)
        end
    end

    β = 1.0
    prev_err = Inf
    for i in 1:max_iter
        @time K_tan, R_int = Sub_LinearProblem(Jac, Res, recipe, Spaces)
        Ap = spzeros(Float64, ndofs, ndofs)
        Fp = zeros(Float64, ndofs)
        min_detJ = Inf
        dim_geo = size(recipe.model.x, 2)
        for sp in values(Spaces)
            for m in sp.Measures
                if !isempty(m.gs)
                    min_detJ = min(min_detJ, minimum(view(m.gs, :, size(m.gs, 2))))
                end
            end
        end
        h_elem = (min_detJ < Inf && min_detJ > 0) ? min_detJ^(1.0 / dim_geo) : 1.0
        α_base = isempty(K_tan.nzval) ? 1e10 : maximum(abs, K_tan) * 100.0
        α = α_base / h_elem
        if !isempty(recipe.Dirichlet_Entities)
            cont_tris = typeof(recipe.Dirichlet_Entities[1])[]
            cont_vals = eltype(recipe.Dirichlet_Values)[]
            cont_masks = Vector{Bool}[]
            point_sets = PointSet[]
            point_vals = eltype(recipe.Dirichlet_Values)[]
            point_masks = Vector{Bool}[]
            # 1. Clasificación al vuelo (0D vs Continuo) guardando la máscara correspondiente
            for (tri, val, mask) in zip(recipe.Dirichlet_Entities, recipe.Dirichlet_Values, recipe.Dirichlet_Masks)
                info, cat = get_entity_info(recipe.model, tri.tag)
                if cat == :nodes
                    # Si el entity es un nodo o conjunto de nodos (devuelve Int o Matrix/Vector)
                    indices = info isa Int ? [info] : info
                    for idx in indices
                        coords = recipe.model.x[idx:idx, :]
                        push!(point_sets, PointSet(coords))
                        push!(point_vals, val)
                        push!(point_masks, mask)
                    end
                else
                    push!(cont_tris, tri)
                    push!(cont_vals, val)
                    push!(cont_masks, mask)
                end
            end
            # 2. Ensamblaje de Entidades Continuas
            if !isempty(cont_tris)
                dsets = map(cont_tris) do tri
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
                for (diri, dΓc, mask) in zip(cont_vals, dΓd, cont_masks)
                    coefs = ntuple(i -> mask[i] ? α : 0.0, Val(D))
                    temp_space = FlexiSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes)
                    if D == 1
                        c = coefs[1]
                        f_pen = δu -> c * diri * δu
                        f_pure_diri = sa -> f_pen(sa).phi
                        op_arg = (δu, u) -> c * δu * u
                    else
                        Mα = SDiagonal{D,Float64}(coefs)
                        f_pen = δu -> δu ⋅ (x -> begin
                            dv = diri isa Function ? diri(x) : diri
                            VectorField(ntuple(k -> Float64(dv[k]) * coefs[k], Val(D))...)
                        end)
                        f_pure_diri = sa -> begin
                            m = f_pen(sa)
                            SVector{D,Float64}(ntuple(i -> m[i].phi, Val(D)))
                        end
                        op_arg = (δu, u) -> Mα * (δu ⋅ u)
                    end
                    Fp .+= Linear_Assembler(f_pure_diri, temp_space)
                    Ap += Bilinear_Assembler(op_arg, temp_space)
                end
            end

            # 3. Ensamblaje de Entidades Puntuales (Nodos)
            if !isempty(point_sets)
                for (pset, diri, mask) in zip(point_sets, point_vals, point_masks)
                    Space_Dp = get(Spaces, pset, nothing)
                    if Space_Dp === nothing
                        Space_Dp = build_point_space(recipe, pset)
                        Spaces[pset] = Space_Dp
                    end
                    dΓp = Space_Dp.Measures[1]
                    temp_space_p = FlexiSpace(Space_Dp.domain, Space_Dp.boundary, Space_Dp.Field_Type, [dΓp], Space_Dp.nnodes)
                    coefs = ntuple(i -> mask[i] ? α : 0.0, Val(D))
                    if D == 1
                        c = coefs[1]
                        f_pen_p = δu -> c * diri * δu
                        f_pure_diri_p = sa -> f_pen_p(sa).phi
                        op_arg_p = (δu, u) -> c * δu * u
                    else
                        Mα = SDiagonal{D,Float64}(coefs)
                        f_pen_p = δu -> δu ⋅ (x -> begin
                            dv = diri isa Function ? diri(x) : diri
                            VectorField(ntuple(k -> Float64(dv[k]) * coefs[k], Val(D))...)
                        end)
                        f_pure_diri_p = sa -> begin
                            m = f_pen_p(sa)
                            SVector{D,Float64}(ntuple(i -> m[i].phi, Val(D)))
                        end
                        op_arg_p = (δu, u) -> Mα * (δu ⋅ u)
                    end
                    Fp .+= Linear_Assembler(f_pure_diri_p, temp_space_p)
                    Ap += Bilinear_Assembler(op_arg_p, temp_space_p)
                end
            end
        end
        K_tot = K_tan + Ap
        R_tot = R_int + (Ap * u_nodal) - Fp
        du = K_tot \ R_tot
        u_nodal .-= β * du
        rel_err = norm(du) / (norm(u_nodal) + 1e-10)
        if rel_err > prev_err
            β = β / 2.0
            if β < 0.05
                β = 1.0
            end
        end
        prev_err = rel_err
        println("Iteración $i - Error_L2: $rel_err | NR_Step: $β")
        if rel_err < tol
            break
        end
        Jac, Res = NL_Op.step_builder(uhs...)
    end
    return Flexifunctions
end