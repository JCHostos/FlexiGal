using FlexiGal
using LinearAlgebra
Domain = (0.16, 1.0); Divisions = (36, 120)
ε = 3.0
clustering_right(x) = [0.16 * tanh(x[1] / 0.16 * ε) / tanh(ε), x[2]]

dm_clustered(x) = [ε / (tanh(ε) * cosh(x[1] / 0.16 * ε)^2), 1.0]

model = create_model(Domain, Divisions; map=clustering_right);
dmax = [1.75, 1.75];
ngpts = 3;

const Tc = 1803.0; const vc = 0.03; const Tw = 303.0; const h = 500;
const kₛ = 30.0; const kₗ = 192.0; const cpₛ = 632.0; const cpₗ = 806.0;
const ρ = 7200.0; const Hf = 272000.0; const Tₗ = 1778; const Tₛ = 1763;

fₛ(T) = T <= Tₛ ? 1.0 : (T >= Tₗ ? 0.0 : 1.0 - (T - Tₛ) / (Tₗ - Tₛ))
q₀(x) = x[2] <= 0.6 ? 2.19e6 - 5.64e5 * sqrt(x[2] / vc) + 3.37e6 * x[2] + 0.16 * x[2]^2 : 0.0
hₛ(x) = x[2] <= 0.6 ? 0.0 : h

shell(x) = x[1] > 0.135; bulk(x)  = x[1] <= 0.135;

set_tag!(model, shell; name="Shell"); set_tag!(model, bulk; name="Bulk");
Ω_fem = Triangulation(model, "Bulk"); Ω_mk = Triangulation(model, "Shell");
Γb = Triangulation(model, "Bottom"); Γᵣ = Triangulation(model, "Right");
dΩ_fem = IntegrationSet(Ω_fem, 2); dΩ_mk = IntegrationSet(Ω_mk, ngpts);
dΓᵣ = IntegrationSet(Γᵣ, ngpts);

Tspace = ApproxSpace(model, [Ω_fem, Ω_mk, Γb, Γᵣ], Float64; dmax,dm_map=dm_clustered, method=[:FEM,:EFG,:FEM,:EFG], technique =:MK,
    Dirichlet_Boundaries=[Γb], Dirichlet_Values=[Tc]);

fspace = ApproxSpace(model, [Ω_fem, Ω_mk, Γb, Γᵣ], Float64; dmax, dm_map=dm_clustered, method=[:FEM,:EFG,:FEM,:EFG], technique =:MK);

const v = VectorField(0.0, vc); r(x) = x[1];

@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T) * r) + δT * (ρ * cpₛ * v ⋅ ∇(T) * r))dΩ_fem + 
∫(∇(δT) ⋅ (kₛ * ∇(T) * r) + δT * (ρ * cpₛ * v ⋅ ∇(T) * r))dΩ_mk + ∫(δT * (hₛ * T * r))dΓᵣ
@WeakForm b(δT) = ∫((-δT * q₀) * r)dΓᵣ + ∫((δT * hₛ * Tw) * r)dΓᵣ
op = Linear_Problem(a, b, Tspace);
Th = Solve(op);

function Deferred_Piccard(Th, op, fspace, Tspace; tol=1e-6, max_iter=60)
    _,_,Spaces=op
    Tspace_Built_fem = Get_space_from_IntegrationSet(Spaces, dΩ_fem)
    Tspace_Built_mk = Get_space_from_IntegrationSet(Spaces, dΩ_mk)
    Measure_fem = Get_Measures(Tspace_Built_fem)
    Measure_mk = Get_Measures(Tspace_Built_mk)
    Th_old = Th; Th_new = Th;
    iter = 0; err = 1.0; erra = 1.0;
    k_nl(T) = fₛ(T) * kₛ + (1 - fₛ(T)) * kₗ
    cp_nl(T) = fₛ(T) * cpₛ + (1 - fₛ(T)) * cpₗ
    β = 1.0; ε = 1e-9;
    Th_fem=Th[dΩ_fem]; Th_mk=Th[dΩ_mk]
    @WeakForm aₚ(δf, f) = ∫(δf * (f * r) + ε * ∇(δf) ⋅ (∇(f) * r))dΩ_fem + ∫(δf * (f * r) + ε * ∇(δf) ⋅ (∇(f) * r))dΩ_mk
    @WeakForm bₚ(δf) = ∫((δf * (fₛ ∘ Th_fem)) * r)dΩ_fem + ∫((δf * (fₛ ∘ Th_mk)) * r)dΩ_mk
    op_fs = Linear_Problem(aₚ, bₚ, fspace, op)
    while err > tol && iter < max_iter
        iter += 1
        fₛh = let Th = Th_old, fspace = fspace, op_fs=op_fs, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
        Th_mk = Th[dΩ_mk]; Th_fem = Th[dΩ_fem]
        @WeakForm bₚ(δf) = ∫((δf * (fₛ ∘ Th_fem)) * r)dΩ_fem + ∫((δf * (fₛ ∘ Th_mk)) * r)dΩ_mk
        op_fs = Reassemble_Vector!(bₚ,fspace,op_fs)
        Solve(op_fs)
        end
        Th_new = let Th_old = Th_old, fₛh = fₛh, Tspace = Tspace, op=op, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
            fₛh_fem=fₛh[dΩ_fem]; fₛh_mk=fₛh[dΩ_mk]
            Th_old_fem=Th_old[dΩ_fem]; Th_old_mk=Th_old[dΩ_mk]
            @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl ∘ Th_old_fem * ∇(T) * r) + δT * ρ * (cp_nl ∘ Th_old_fem) * ((v ⋅ ∇(T)) * r))dΩ_fem + 
            ∫(∇(δT) ⋅ (k_nl ∘ Th_old_mk * ∇(T) * r) + δT * ρ * (cp_nl ∘ Th_old_mk) * ((v ⋅ ∇(T)) * r))dΩ_mk  + ∫(δT * (hₛ * T * r))dΓᵣ
            @WeakForm b_picard(δT) = ∫((-δT * q₀) * r)dΓᵣ + ∫((δT * hₛ * Tw) * r)dΓᵣ + 
            ∫((δT * ρ * Hf * v ⋅ ∇(fₛh_fem)) * r)dΩ_fem + ∫((δT * ρ * Hf * v ⋅ ∇(fₛh_mk)) * r)dΩ_mk
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
    return Th_new
end
Th2 = Deferred_Piccard(Th, op, fspace, Tspace; max_iter=150, tol=1e-5);