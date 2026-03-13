using FlexiGal
using LinearAlgebra
Domain = (0.16, 1.0)
Divisions = (30, 80)
dmax = 1.75
model = create_model(Domain, Divisions);
dm = Influence_Domains(model, Domain, Divisions, dmax);
ngpts = 3;
const Tc = 1803.0;
const vc = 0.03;
const Tw = 303.0;
const h = 500;
const kₛ = 30.0;
const kₗ = 192.0;
const cpₛ = 632.0
const cpₗ = 806.0;
const ρ = 7200.0;
const Hf = 272000.0;
const Tₗ = 1783;
const Tₛ = 1763;
function fₛ(T)
    if T <= Tₛ
        fₛ = 1.0
    elseif T > Tₛ && T < Tₗ
        fₛ = 1.0 - (T - Tₛ) / (Tₗ - Tₛ)
    elseif T >= Tₗ
        fₛ = 0.0
    end
    return fₛ
end
function q₀(x)
    y = x[2]
    if y <= 0.6
        val = 2.19e6 - 5.64e5 * sqrt(y / vc) + 3.37e6 * y + 0.16 * y^2
    else
        val = 0.0
    end
    return val
end

function hₛ(x)
    return x[2] <= 0.6 ? 0.0 : h
end
Ω = Triangulation(model, "Domain");
Γb = Triangulation(model, "Bottom");
Γr = Triangulation(model, "Right");
dΩ = IntegrationSet(Ω, ngpts);
dΓr = IntegrationSet(Γr, ngpts);
Tspace = ApproxSpace(model, [Ω, Γb, Γr], Float64, dmax;
    Dirichlet_Boundaries=[Γb], Dirichlet_Values=[Tc]);
fspace = ApproxSpace(model, [Ω, Γb, Γr], Float64, dmax);
const v = VectorField(0.0, vc)
r(x) = x[1]
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T) * r) + δT * (ρ * cpₛ * v ⋅ ∇(T) * r))dΩ + ∫(δT * (hₛ * T * r))dΓr
@WeakForm b(δT) = ∫((-δT * q₀) * r)dΓr + ∫((δT * hₛ * Tw) * r)dΓr
op = Linear_Problem(a, b, Tspace);
Th = Solve(op, dΩ);
_, _, Spaces = op;
function Deferred_Piccard(Th, Spaces, fspace, Tspace; tol = 1e-4, max_iter = 60)
    Tspace_Built = Get_space_from_IntegrationSet(Spaces,dΩ)
    Measure = Get_Measures(Tspace_Built)
    Th_old = Th
    Th_new = Th
    iter = 0
    err = 1.0
    erra = 1.0;
    k_nl(T) = fₛ(T) * kₛ + (1 - fₛ(T)) * kₗ
    cp_nl(T) = fₛ(T) * cpₛ + (1 - fₛ(T)) * cpₗ
    β=1.0
    while err > tol && iter < max_iter
        iter += 1 # Ya no necesitas 'global iter'
        fₛh = let Th = Th_old, Spaces = Spaces, fspace = fspace
            @WeakForm aₚ(δf, f) = ∫(δf * (f * r))dΩ
            @WeakForm bₚ(δf) = ∫((δf * (fₛ ∘ Th)) * r)dΩ
            op_fs = Linear_Problem(aₚ, bₚ, fspace, Spaces)
            Solve(op_fs, dΩ)
        end
        Th_new = let Th_old = Th_old, fₛh = fₛh, Spaces = Spaces, Tspace = Tspace
            @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl∘Th_old *∇(T) * r) + δT * ρ * (cp_nl ∘ Th_old) * ((v ⋅ ∇(T)) * r))dΩ + ∫(δT * (hₛ * T * r))dΓr
            @WeakForm b_picard(δT) = ∫((-δT * q₀) * r)dΓr + ∫((δT * hₛ * Tw) * r)dΓr + ∫((δT * ρ * Hf * v ⋅ ∇(fₛh)) * r)dΩ
            @time op_nl = Linear_Problem(a_picard, b_picard, Tspace, Spaces)
            Solve(op_nl, dΩ)
        end
        T_new = Get_Nodal_Values(Th_new)
        T_old = Get_Nodal_Values(Th_old)
        err = norm(T_new - T_old) / (norm(T_new)) 
        println("Iteración Picard $iter: Error Relativo = $err")
        if err>erra
        β=β/2
        T_old = T_old.*(1-β) + T_new.*β
        else
        erra=err;
        T_old = T_old.*(1-β) + T_new.*β
        end
        Th_old = EFGFunction(T_old,Tspace_Built,Measure[1])
    end
    if err <= tol
        println("--- Convergencia exitosa en $iter iteraciones ---")
    else
        println("--- Advertencia: No hubo convergencia (Error: $err) ---")
    end 
    return Th_new
end
Th2 = Deferred_Piccard(Th, Spaces, fspace, Tspace; max_iter = 80);