function Get_space_from_IntegrationSet(Spaces::Dict{IntegrationSet,EFGSpace}, target_iset::IntegrationSet)
    return get(Spaces, target_iset) do
        error("No se encontró espacio para el set con tag: '$(target_iset.tri.tag)'")
    end
end

Get_Measures(a::EFGSpace) = a.Measures

function extract_EFGFunction(field)
    if field isa EFGFunction
        return field
    end
    if !isprimitivetype(typeof(field))
        for name in fieldnames(typeof(field))
            child = getproperty(field, name)
            result = extract_EFGFunction(child)
            if result isa EFGFunction
                return result
            end
        end
    end
    return nothing
end
function Solve(op, iset::IntegrationSet)
    A, F, Spaces = op
    u_nodal = A \ F
    Space = Get_space_from_IntegrationSet(Spaces, iset)
    M = Space.Measures isa AbstractVector ? Space.Measures[1] : Space.Measures
    SolvedEFGfunction = EFGFunction(u_nodal, Space, M) 
    return SolvedEFGfunction;
end

function Solve(op)
    A, F, Spaces = op
    u_nodal = A \ F
    EFGfunctions = Dict(s => EFGFunction(u_nodal, sp, sp.Measures[1]) for (s, sp) in Spaces)
    return EFGfunctions;
end

struct NonLinearOperator
    step_builder::Function
    space::ApproxSpace
end

function NLSolver(Jacobian, Residual, recipe::ApproxSpace, iset::IntegrationSet; u_seed=nothing, tol=1e-6, max_iter=15)
    res_jac = Jacobian
    res_res = Residual
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    u_nodal = (u_seed === nothing) ? zeros(Float64, ndofs) : copy(u_seed)
    max_deg = 1
    Spaces = Dict{IntegrationSet,EFGSpace}()
    if res_jac isa Integrated
        isets = [res_jac.b]
        max_deg = max(max_deg, res_jac.b.degree)
        if !haskey(Spaces, res_jac.b)
            println("Constructing new")
            Space = build_space(recipe, isets)
            for s in isets; Spaces[s] = Space; end
        else
            Space = Spaces[res_jac.b]
        end
    elseif res_jac isa MultiIntegrated
        for ii in 1:length(res_jac.terms)
            term = res_jac.terms[ii]
            isets = [term.b]
            max_deg = max(max_deg, term.b.degree)
            if !haskey(Spaces, term.b)
            println("Constructing new")
            Space = build_space(recipe, isets)
            for s in isets; Spaces[s] = Space; end
            else
            Space = Spaces[term.b]
            end
        end
    end
    if res_res isa Integrated
        isets = [res_res.b]
        max_deg = max(max_deg, res_res.b.degree)
           if !haskey(Spaces, res_res.b)
                println("Constructing new")
                Space = build_space(recipe, isets)
                for s in isets
                    Spaces[s] = Space
                end
           else
                Space = Spaces[res_res.b]
           end
    elseif res_res isa MultiIntegrated
        for ii in 1:length(res_res.terms)
            term = res_res.terms[ii]
            isets = [term.b]
            max_deg = max(max_deg, term.b.degree)
            if !haskey(Spaces, term.b)
                println("Constructing new")
                Space = build_space(recipe, isets)
                for s in isets
                    Spaces[s] = Space
                end
            else
                Space = Spaces[term.b]
            end
        end
    end
    Ap = spzeros(Float64, ndofs, ndofs)
    Fp = zeros(Float64, ndofs)
    α = 1e6
    if !isempty(recipe.Dirichlet_Boundaries)
        dsets = [IntegrationSet(tri, max_deg) for tri in recipe.Dirichlet_Boundaries]
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
            dirichlet_values = recipe.Dirichlet_Values
            Fp = zeros(ndofs)
            for (diri, dΓc) in zip(dirichlet_values, dΓd)
                f_pen = if D == 1
                    δu -> (α * (diri * δu))
                else
                    δu -> (δu ⋅ (diri * α))
                end
                f_pure = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1
                        return m.phi
                    elseif D == 2
                        return SVector{2,Float64}(m[1].phi, m[2].phi)
                    elseif D == 3
                        return SVector{3,Float64}(m[1].phi, m[2].phi, m[3].phi)
                    end
                end
                Fp .+= Linear_Assembler(f_pure, EFGSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes))
            end
        end
        op_arg =
            if D == 1
                (δu, u) -> (α * δu * u)
            else
                (δu, u) -> (α * (δu ⋅ u))
            end
        f_pure = (sa, sb) -> op_arg(sa, sb)
        Ap = Bilinear_Assembler(f_pure, Space_D)
    end

    EFGfunctions = Dict(s => EFGFunction(u_nodal, sp, sp.Measures[1]) for (s, sp) in Spaces)
    
    for i in 1:max_iter
        @time K_tan, R_int = Sub_LinearProblem(Jacobian, Residual, recipe, Spaces)
        K_tot = K_tan + Ap
        R_tot = R_int + (Ap * u_nodal)-Fp
        du = K_tot \ R_tot
        u_nodal .-= du
        rel_err = norm(du) / (norm(u_nodal) + 1e-10)
        println("Error: $rel_err")
        if rel_err < tol
           break
        end
        #EFGfunctions = Dict(m => EFGFunction(u_nodal, Space, m) for m in Space.Measures)
    end
    Space = Get_space_from_IntegrationSet(Spaces, iset)
    Solved_EFGfunctions = [EFGFunction(u_nodal, Space, m) for m in Space.Measures] #Esto sí quiero que salga como vector
    return length(Solved_EFGfunctions) == 1 ? Solved_EFGfunctions[1] : Solved_EFGfunctions
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

function Prueba_Macro(NL_Op; u_seed=nothing, tol=1e-6, max_iter=15)
    recipe = NL_Op.space
    D = field_dim(recipe.Field_Type)
    nnodes = recipe.model.nnodes
    ndofs = D * nnodes
    u_nodal = (u_seed === nothing) ? zeros(Float64, ndofs) : copy(u_seed)
    max_deg = 1
    m = methods(NL_Op.step_builder)
    n_args = length(Base.method_argnames(last(collect(m)))) - 1
    arg_isets = Vector{IntegrationSet}(undef, n_args)

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
                    encontrado = true; break
                end
            end
            encontrado && break
        end
        !encontrado && error("No se detectó IntegrationSet para el argumento $i")
    end
    Spaces = Dict{IntegrationSet, EFGSpace}()
    isets_vec = unique(arg_isets)
    for s in isets_vec
        Spaces[s] = build_space(recipe, [s])
        max_deg = max(max_deg, s.degree)
    end
    EFGfunctions = Dict(s => EFGFunction(u_nodal, sp, sp.Measures[1]) for (s, sp) in Spaces)
    uhs = [EFGfunctions[s] for s in isets_vec] 
    Jac, Res = NL_Op.step_builder(uhs...)
    if Jac isa Integrated
        isets = [Jac.b]
        max_deg = max(max_deg, Jac.b.degree)
        if !haskey(Spaces, Jac.b)
            println("Constructing new")
            Space = build_space(recipe, isets)
            for s in isets; Spaces[s] = Space; end
        end
    elseif Jac isa MultiIntegrated
        for ii in 1:length(Jac.terms)
            term = Jac.terms[ii]
            isets = [term.b]
            max_deg = max(max_deg, term.b.degree)
            if !haskey(Spaces, term.b)
            println("Constructing new")
            Space = build_space(recipe, isets)
            for s in isets; Spaces[s] = Space; end
            end
        end
    end
    if Res isa Integrated
        isets = [Res.b]
        max_deg = max(max_deg, Res.b.degree)
           if !haskey(Spaces, Res.b)
                println("Constructing new")
                Space = build_space(recipe, isets)
                for s in isets
                    Spaces[s] = Space
                end
           end
    elseif Res isa MultiIntegrated
        for ii in 1:length(Res.terms)
            term = Res.terms[ii]
            isets = [term.b]
            max_deg = max(max_deg, term.b.degree)
            if !haskey(Spaces, term.b)
                println("Constructing new")
                Space = build_space(recipe, isets)
                for s in isets
                    Spaces[s] = Space
                end
            end
        end
    end
    Ap = spzeros(Float64, ndofs, ndofs)
    Fp = zeros(Float64, ndofs)
    α = 1e6
    if !isempty(recipe.Dirichlet_Boundaries)
        dsets = [IntegrationSet(tri, max_deg) for tri in recipe.Dirichlet_Boundaries]
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
            dirichlet_values = recipe.Dirichlet_Values
            Fp = zeros(ndofs)
            for (diri, dΓc) in zip(dirichlet_values, dΓd)
                f_pen = if D == 1
                    δu -> (α * (diri * δu))
                else
                    δu -> (δu ⋅ (diri * α))
                end
                f_pure = (sa) -> begin
                    m = f_pen(sa)
                    if D == 1
                        return m.phi
                    elseif D == 2
                        return SVector{2,Float64}(m[1].phi, m[2].phi)
                    elseif D == 3
                        return SVector{3,Float64}(m[1].phi, m[2].phi, m[3].phi)
                    end
                end
                Fp .+= Linear_Assembler(f_pure, EFGSpace(Space_D.domain, Space_D.boundary, Space_D.Field_Type, [dΓc], Space_D.nnodes))
            end
        end
        op_arg =
            if D == 1
                (δu, u) -> (α * δu * u)
            else
                (δu, u) -> (α * (δu ⋅ u))
            end
        f_pure = (sa, sb) -> op_arg(sa, sb)
        Ap = Bilinear_Assembler(f_pure, Space_D)
    end
    β = 1.0
    prev_err = Inf
    for i in 1:max_iter
        @time K_tan, R_int = Sub_LinearProblem(Jac, Res, recipe, Spaces)
        K_tot = K_tan + Ap
        R_tot = R_int + (Ap * u_nodal)-Fp
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
    return EFGfunctions 
end