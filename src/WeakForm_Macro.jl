macro WeakForm(args...)
    expr = args[end] 
    if !(expr isa Expr) || expr.head != :(=) || expr.args[1].head != :call
        error("Uso incorrecto: @WeakForm debe preceder a una función, ej: @WeakForm a(u, v) = ...")
    end
    name = expr.args[1].args[1]
    func_args = expr.args[1].args[2:end] # Cambié el nombre a func_args para no confundir
    body = expr.args[2]
    function transform_integrals!(ex)
        if ex isa Expr
            if ex.head == :call && ex.args[1] == :∫
                inner_math = ex.args[2]
                ex.args[2] = :(($(func_args...),) -> $inner_math)
            end
            for i in eachindex(ex.args)
                ex.args[i] = transform_integrals!(ex.args[i])
            end
        end
        return ex
    end
    new_body = transform_integrals!(body)
    return esc(:($name = $new_body))
end

macro NL_WeakForm(u_var, body)
    if !(body isa Expr && body.head == :block)
        body = Expr(:block, body)
    end
    active_doms = Symbol[]
    function find_active_domains(ex)
        if ex isa Expr
            if ex.head == :call && ex.args[1] == :*
                for i in 2:length(ex.args)-1
                    term = ex.args[i]
                    next_term = ex.args[i+1]
                    if term isa Expr && term.head == :call && term.args[1] == :∫ && next_term isa Symbol
                        contains_u = false
                        function check_u(e)
                            if e == u_var contains_u = true end
                            if e isa Expr; for a in e.args; check_u(a) end end
                        end
                        check_u(term.args[2])
                        if contains_u
                            push!(active_doms, next_term)
                        end
                    end
                end
            end
            for a in ex.args; find_active_domains(a); end
        end
    end
    find_active_domains(body)
    unique_active = unique(active_doms)
    N = length(unique_active)
    if N == 0
        error("La variable $u_var no parece ser usada en ninguna integral ∫(...) del bloque.")
    end
    u_args = [Symbol("u_", i) for i in 1:N]
    func_expr = quote
        (($(u_args...),)) -> begin
            all_jacs = []
            all_res  = []
            function _filter_by_domain(integrated_obj, target_dom)
                if integrated_obj === nothing return nothing end
                if hasproperty(integrated_obj, :terms)
                    valid = filter(t -> t.b === target_dom, integrated_obj.terms)
                    if isempty(valid) return nothing end
                    res = valid[1]
                    for i in 2:length(valid); res = res + valid[i]; end
                    return res
                else
                    return integrated_obj.b === target_dom ? integrated_obj : nothing
                end
            end
            $((quote
                let $(u_var) = $(u_args[i])
                    local Jac, Res
                    $(body)
                    push!(all_jacs, Jac)
                    push!(all_res, Res)
                end
            end for i in 1:N)...)
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