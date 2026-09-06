using FlexiGal
using LinearAlgebra
using NLsolve
Domain = (1.0, 0.1); Divisions = (320, 2)
model = create_model(Domain, Divisions);
dmax = [1.5, 1.05];
ngpts = 3;

const T0 = 940.0; const Pe = 7; const TL = 298.0;
const kₛ = 231.0; const kₗ = 90.7; const cpₛ = 900.0; const cpₗ = 1300.0;
const ρ = 2700.0; const Hf = 396400.0; const Tₗ = 930.0; const Tₛ = 919.0;
vc= Pe*kₗ/(ρ*cpₗ)

fₛ(T) = T <= Tₛ ? 1.0 : (T >= Tₗ ? 0.0 : 1.0 - (T - Tₛ) / (Tₗ - Tₛ))

MK(x) = x[1] >0.5 && x[1]<0.75; FEM(x)  = !MK(x);

set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
Ω_fem = Triangulation(model, "FEM"); Ω_mk = Triangulation(model, "MK");
Γ0 = Triangulation(model, "Left"); ΓL = Triangulation(model, "Right");
dΩ_fem = IntegrationSet(Ω_fem, 3); dΩ_mk = IntegrationSet(Ω_mk, ngpts);

Tspace = ApproxSpace(model, [Ω_fem, Ω_mk, Γ0, ΓL], Float64; dmax, method=[:FEM,:FEM,:FEM,:FEM], technique =:MK,
    Dirichlet_Boundaries=[Γ0,ΓL], Dirichlet_Values=[T0,TL]);

fspace = ApproxSpace(model, [Ω_fem, Ω_mk, Γ0, ΓL], Float64; dmax, method=[:FEM,:FEM,:FEM,:FEM], technique =:MK);

const v = VectorField(vc, 0.0);

@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρ * cpₛ * v ⋅ ∇(T)))dΩ_fem + 
∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρ * cpₛ * v ⋅ ∇(T)))dΩ_mk
op = Linear_Problem(a,Tspace);
Th = Solve(op);

function Deferred_Piccard(Th, op, fspace, Tspace; tol=1e-6, max_iter=60)
    _,_,Spaces=op
    Tspace_Built_fem = Get_space_from_IntegrationSet(Spaces, dΩ_fem)
    Tspace_Built_mk = Get_space_from_IntegrationSet(Spaces, dΩ_mk)
    Measure_fem = Get_Measures(Tspace_Built_fem)
    Measure_mk = Get_Measures(Tspace_Built_mk)
    Th_old = Th; Th_new = Th;
    fₛh = nothing
    iter = 0; err = 1.0; erra = 1.0;
    k_nl(T) = fₛ(T) * kₛ + (1 - fₛ(T)) * kₗ
    cp_nl(T) = fₛ(T) * cpₛ + (1 - fₛ(T)) * cpₗ
    β = 1.0; ε = 1e-9;
    Th_fem=Th[dΩ_fem]; Th_mk=Th[dΩ_mk]
    @WeakForm aₚ(δf, f) = ∫(δf * f + ε * ∇(δf) ⋅ (∇(f)))dΩ_fem + ∫(δf * f + ε * ∇(δf) ⋅ ∇(f))dΩ_mk
    @WeakForm bₚ(δf) = ∫(δf * (fₛ ∘ Th_fem))dΩ_fem + ∫(δf * (fₛ ∘ Th_mk))dΩ_mk
    op_fs = Linear_Problem(aₚ, bₚ, fspace, op)
    while err > tol && iter < max_iter
        iter += 1
        fₛh = let Th = Th_old, fspace = fspace, op_fs=op_fs, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
        Th_mk = Th[dΩ_mk]; Th_fem = Th[dΩ_fem]
        @WeakForm bₚ(δf) = ∫(δf * (fₛ ∘ Th_fem))dΩ_fem + ∫(δf * (fₛ ∘ Th_mk))dΩ_mk
        op_fs = Reassemble_Vector!(bₚ,fspace,op_fs)
        Solve(op_fs)
        end
        Th_new = let Th_old = Th_old, fₛh = fₛh, Tspace = Tspace, op=op, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
            fₛh_fem=fₛh[dΩ_fem]; fₛh_mk=fₛh[dΩ_mk]
            Th_old_fem=Th_old[dΩ_fem]; Th_old_mk=Th_old[dΩ_mk]
            @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl ∘ Th_old_fem * ∇(T)) + δT * ρ * (cp_nl ∘ Th_old_fem) * (v ⋅ ∇(T)))dΩ_fem + 
            ∫(∇(δT) ⋅ (k_nl ∘ Th_old_mk * ∇(T)) + δT * ρ * (cp_nl ∘ Th_old_mk) * (v ⋅ ∇(T)))dΩ_mk
            @WeakForm b_picard(δT) = ∫(δT * ρ * Hf * v ⋅ ∇(fₛh_fem))dΩ_fem + ∫(δT * ρ * Hf * v ⋅ ∇(fₛh_mk))dΩ_mk
            @time op_nl = Linear_Problem(a_picard, b_picard, Tspace, op)
            Solve(op_nl)
        end
        T_new = Get_Nodal_Values(Th_new[dΩ_fem])
        T_old = Get_Nodal_Values(Th_old[dΩ_fem])
        err = norm(T_new - T_old) / (norm(T_new))
        println("Picard Iteration $iter: Relative Error = $err")
        err > erra ? β /= 2 : erra = err
        @. T_old = T_old * (1 - β) + T_new * β
        β < 0.05 && (β = 1.0)
        Th_old = Dict(dΩ_fem => FlexiFunction(T_old, Tspace_Built_fem, Measure_fem[1]),
        dΩ_mk  => FlexiFunction(T_old, Tspace_Built_mk, Measure_mk[1]))
    end
    return Th_new, fₛh
end
Th2,fₛh = Deferred_Piccard(Th, op, fspace, Tspace; max_iter=150, tol=1e-5);

# ==============================================================================
# POST-PROCESAMIENTO Y COMPARACIÓN CON LA SOLUCIÓN ANALÍTICA EXACTA
# ==============================================================================

function exact_solution_at_nodes(
    x_nodes;
    ks=kₛ, kl=kₗ, Cps=cpₛ, Cpl=cpₗ,
    Tsol=Tₛ, Tliq=Tₗ, rho=ρ,
    To=T0, TL=TL, vc=vc, Hf=Hf, L=1.0
)
    # Propiedades efectivas de la zona pastosa
    Teq = (Tsol + Tliq) / 2.0
    fseq = (Teq - Tliq) / (Tsol - Tliq)
    dfseqdT = 1.0 / (Tsol - Tliq)
    fssol = 1.0
    fsliq = 0.0

    km = fseq * ks + (1.0 - fseq) * kl
    Cpm = fseq * Cps + (1.0 - fseq) * Cpl

    alpham = km / (rho * (Cpm - Hf * dfseqdT))
    alphas = ks / (Cps * rho)
    alphal = kl / (Cpl * rho)
    vo = vc

    # Sistema no lineal estricto de Alvarez
    function stefan_mushy!(F, S)
        # S[1] = Sl (Liquidus), S[2] = Ss (Solidus)
        F[1] = (Tsol - Tliq) * (Cpm - Hf * dfseqdT) * exp(vo * S[1] / alpham) / (exp(vo * S[2] / alpham) - exp(vo * S[1] / alpham)) - 
               (Tliq - To) * Cpl * exp(vo * S[1] / alphal) / (exp(vo * S[1] / alphal) - 1.0) + 
               Hf * fsliq

        F[2] = (TL - Tsol) * Cps * exp(vo * S[2] / alphas) / (exp(vo * L / alphas) - exp(vo * S[2] / alphas)) - 
               (Tsol - Tliq) * (Cpm - Hf * dfseqdT) * exp(vo * S[2] / alpham) / (exp(vo * S[2] / alpham) - exp(vo * S[1] / alpham)) + 
               Hf * (1.0 - fssol)  # ¡Acá estaba el pifia! (1.0 - 1.0) = 0.0
    end

    res = nlsolve(stefan_mushy!, [0.81, 0.83])
    Sl, Ss = res.zero[1], res.zero[2]

    # Evaluación en la coordenada x de cada nodo
    T_exact = zeros(length(x_nodes))
    for (i, x) in enumerate(x_nodes)
        if x <= Sl
            T_exact[i] = (Tliq - To) / (exp(vo * Sl / alphal) - 1.0) * (exp(vo * x / alphal) - 1.0) + To
        elseif x <= Ss
            T_exact[i] = (Tsol - Tliq) / (exp(vo * Ss / alpham) - exp(vo * Sl / alpham)) * (exp(vo * x / alpham) - exp(vo * Sl / alpham)) + Tliq
        else
            T_exact[i] = (TL - Tsol) / (exp(vo * L / alphas) - exp(vo * Ss / alphas)) * (exp(vo * x / alphas) - exp(vo * Ss / alphas)) + Tsol
        end
    end

    return T_exact, Sl, Ss
end

# 1. Coordenada x de los nodos (primera columna de la matriz 303x2)
x_nodes = model.x[:, 1]

# 2. Valores nodales numéricos
T_num = Get_Nodal_Values(Th2[dΩ_fem])

# 3. Solución analítica evaluada en x
T_exact, Sl_exact, Ss_exact = exact_solution_at_nodes(x_nodes)

# 4. Error relativo L2
Thₑfem=FlexiFunction(T_exact,Tspace,dΩ_fem);
Thₑmk=FlexiFunction(T_exact,Tspace,dΩ_mk);
dThfem=Thₑfem-Th2[dΩ_fem]
dThmk =Thₑmk-Th2[dΩ_mk]
num=∫(dThfem*dThfem)dΩ_fem + ∫(dThmk*dThmk)dΩ_mk
den = ∫(Thₑfem*Thₑfem)dΩ_fem + ∫(Thₑmk*Thₑmk)dΩ_mk
error_L2i=sqrt(num/den)
error_L2 = norm(T_num - T_exact) / norm(T_exact)

println("\n==================================================")
println(" COMPARACIÓN NUMÉRICA VS ANALÍTICA")
println("==================================================")
println(" Error relativo L2: ", error_L2i)
println(" Frente Liquidus Analítico (Sl): ", Sl_exact)
println(" Frente Solidus Analítico  (Ss): ", Ss_exact)
println("==================================================\n")
