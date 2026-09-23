macro WeakForm(args...)
    expr = args[end]
    if !(expr isa Expr) || expr.head != :(=) || expr.args[1].head != :call
        error("Uso incorrecto: @WeakForm debe preceder a una función, ej: @WeakForm a(u, v) = ...")
    end
    name = expr.args[1].args[1]
    func_args = expr.args[1].args[2:end]
    body = expr.args[2]

    # NUEVO: aplanar patrones de tupla (multifísico: a((δu,δp),(u,p)) = ...)
    function _flatten_vars(a)
        if a isa Symbol
            return Symbol[a]
        elseif a isa Expr && a.head == :tuple
            return vcat([_flatten_vars(x) for x in a.args]...)
        else
            return Symbol[]
        end
    end
    flat_vars = vcat([_flatten_vars(a) for a in func_args]...)

    _unsafe_heads = (:->, :function, :do, :generator, :comprehension, :quote, :macrocall)
    function _contains_var(ex, vars)
        if ex isa Symbol
            return ex in vars
        elseif ex isa Expr
            return any(_contains_var(a, vars) for a in ex.args)
        else
            return false
        end
    end
    bindings = Expr[]
    function _hoist!(ex, vars)
        if ex isa Expr
            if ex.head in _unsafe_heads
                return ex
            elseif ex.head == :call
                if !_contains_var(ex, vars)
                    g = gensym(:hoisted)
                    push!(bindings, :($g = $ex))
                    return g
                else
                    return Expr(:call, [_hoist!(a, vars) for a in ex.args]...)
                end
            else
                return Expr(ex.head, [_hoist!(a, vars) for a in ex.args]...)
            end
        else
            return ex
        end
    end
    function transform_integrals!(ex)
        if ex isa Expr
            if ex.head == :call && ex.args[1] in (:∫, :δ)
                inner_math = ex.args[2]
                hoisted_math = _hoist!(inner_math, flat_vars)   # <- único cambio real: flat_vars en vez de func_args
                ex.args[2] = :(($(func_args...),) -> $hoisted_math)   # aquí sigue func_args intacto, para no romper la destructuración
            end
            for i in eachindex(ex.args)
                ex.args[i] = transform_integrals!(ex.args[i])
            end
        end
        return ex
    end
    new_body = transform_integrals!(body)
    full = quote
        $(bindings...)
        $name = $new_body
    end
    return esc(full)
end

macro NL_WeakForm(u_var, body)
    if !(body isa Expr && body.head == :block)
        body = Expr(:block, body)
    end
    active_doms = Symbol[]
    deps = Set{Symbol}([u_var])
    function _contains_any(e)
        if e isa Symbol
            return e in deps
        elseif e isa Expr
            return any(_contains_any(a) for a in e.args)
        else
            return false
        end
    end

    function update_deps!(e)
        changed = false

        function _extract_fname(sig)
            if sig isa Expr && sig.head == :where
                return _extract_fname(sig.args[1])
            elseif sig isa Expr && sig.head == :call
                return sig.args[1] isa Symbol ? sig.args[1] : nothing
            else
                return nothing
            end
        end

        function visit(x)
            if x isa Expr
                if x.head == :(=) && x.args[1] isa Symbol
                    if !in(x.args[1], deps) && _contains_any(x.args[2])
                        push!(deps, x.args[1])
                        changed = true
                    end
                elseif x.head == :(=) && x.args[1] isa Expr && x.args[1].head == :call
                    fname = x.args[1].args[1]
                    if fname isa Symbol && !in(fname, deps) && _contains_any(x.args[2])
                        push!(deps, fname)
                        changed = true
                    end
                elseif x.head == :function
                    fname = _extract_fname(x.args[1])
                    fbody = x.args[2]
                    if fname isa Symbol && !in(fname, deps) && _contains_any(fbody)
                        push!(deps, fname)
                        changed = true
                    end
                end
                for a in x.args
                    visit(a)
                end
            end
        end
        visit(e)
        return changed
    end
    while update_deps!(body)
    end

    function find_active_domains(ex)
        if ex isa Expr
            if ex.head == :call && ex.args[1] == :*
                for i in 2:length(ex.args)-1
                    term = ex.args[i]
                    next_term = ex.args[i+1]
                    if term isa Expr && term.head == :call && term.args[1] in (:∫, :δ) && next_term isa Symbol
                        if _contains_any(term.args[2])
                            push!(active_doms, next_term)
                        end
                    end
                end
            end
            for a in ex.args
                find_active_domains(a)
            end
        end
    end
    find_active_domains(body)

    unique_active = unique(active_doms)
    N = length(unique_active)
    if N == 0
        error("La variable $u_var no parece ser usada (ni siquiera indirectamente) en ninguna integral ∫(...) del bloque.")
    end

    u_args = [Symbol("u_", i) for i in 1:N]
    func_expr = quote
        (($(u_args...),)) -> begin
            all_jacs = []
            all_res = []
            function _filter_by_domain(integrated_obj, target_dom)
                if integrated_obj === nothing
                    return nothing
                end
                if hasproperty(integrated_obj, :terms)
                    valid = filter(t -> t.b === target_dom, integrated_obj.terms)
                    if isempty(valid)
                        return nothing
                    end
                    res = valid[1]
                    for i in 2:length(valid)
                        res = res + valid[i]
                    end
                    return res
                else
                    return integrated_obj.b === target_dom ? integrated_obj : nothing
                end
            end
            $((
                quote
                    let $(u_var) = $(u_args[i])
                        local Jac, Res
                        $(body)
                        push!(all_jacs, Jac)
                        push!(all_res, Res)
                    end
                end for i in 1:N
            )...)
            dom_vals = [$(unique_active...)]
            Jac_final = nothing
            Res_final = nothing
            for i in 1:$N
                f_jac = _filter_by_domain(all_jacs[i], dom_vals[i])
                f_res = _filter_by_domain(all_res[i], dom_vals[i])
                if f_jac !== nothing
                    Jac_final = (Jac_final === nothing) ? f_jac : (Jac_final + f_jac)
                end
                if f_res !== nothing
                    Res_final = (Res_final === nothing) ? f_res : (Res_final + f_res)
                end
            end
            let $(u_var) = $(u_args[1])
                local Jac, Res
                $(body)
                if Res isa FlexiGal.MultiIntegrated
                    for term in Res.terms
                        if !(term.b in dom_vals)
                            Res_final = (Res_final === nothing) ? term : (Res_final + term)
                        end
                    end
                elseif !(Res.b in dom_vals)
                    Res_final = (Res_final === nothing) ? Res : (Res_final + Res)
                end
            end
            return Jac_final, Res_final
        end
    end
    return esc(func_expr)
end