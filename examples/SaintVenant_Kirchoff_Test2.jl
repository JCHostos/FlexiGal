using FlexiGal
Lx=5.0
Ly=1.0
Domain = (Lx, Ly)
Divisions = (100, 50)
dmax = [2.05,2.05];
const E=1000.0;
const ν=0.3;
const G=E/(2*(1+ν));
const λ=E*ν/((1+ν)*(1-2*ν));
function clustering(x)
    eps_val = 3.0
    nx = Lx * (1.0 - tanh(eps_val * (1.0 - x[1] / Lx)) / tanh(eps_val))
    ny = Ly/2 * tanh((2.0*x[2]-Ly)*eps_val)/tanh(eps_val)
    return [nx, ny]
end
function dm_clustered(x0)
    eps_val=3.0
    s1 = eps_val / (tanh(eps_val) * cosh(eps_val * (1.0 - x0[1] / Lx))^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
const P=-1.0
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=clustering)
ngpts = 3
MK(x)=x[1]<1.0
FEM(x) = !MK(x)
set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
Ω_fem = Triangulation(model, "FEM"); Ω_mk = Triangulation(model, "MK");
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
dΩ_fem = IntegrationSet(Ω_fem, ngpts-1); dΩ_mk = IntegrationSet(Ω_mk, ngpts);
dΓ2 = IntegrationSet(Γ2, ngpts) 
const u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
#const u₀=VectorField(0.0, 0.0)
Uspace = ApproxSpace(model, [Ω_mk,Ω_fem, Γ1, Γ2], VectorField{2,Float64}; dmax, dm_map = dm_clustered, method=[:EFG,:FEM,:EFG,:FEM], technique=:LME, shape=:rectangular, Dirichlet_Boundaries=[Γ1],Dirichlet_Values=[u₀])
const s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))

NL_Problem = @NL_WeakForm u begin
    Egreen(u) = 1/2*(∇(u) + ∇(u)' + (∇(u)')*∇(u))
    dEgreen(u, du) = 1/2*(∇(du) + ∇(du)' + (∇(du)')*∇(u) + (∇(u)')*∇(du))
    Sₛ = 2*G*Egreen(u)
    Sᵥ = λ*tr(Egreen(u))*Id
    dSₛ(du) = 2*G*dEgreen(u, du)
    dSᵥ(du) = λ*tr(dEgreen(u, du))*Id
    @WeakForm Jac(δu, du) = ∫(∇(δu)⊙((Sₛ+Sᵥ)*∇(du)) + dEgreen(u,δu)⊙(dSₛ(du)+dSᵥ(du)))dΩ_mk + ∫(∇(δu)⊙((Sₛ+Sᵥ)*∇(du)) + dEgreen(u,δu)⊙(dSₛ(du)+dSᵥ(du)))dΩ_fem
    @WeakForm Res(δu)     = ∫(dEgreen(u,δu)⊙(Sₛ+Sᵥ))dΩ_mk + ∫(dEgreen(u,δu)⊙(Sₛ+Sᵥ))dΩ_fem + ∫(-δu ⋅ s)dΓ2
end

op = NonLinearOperator(NL_Problem, Uspace);

uh = NL_Solver(op; tol = 1e-4, max_iter=50);