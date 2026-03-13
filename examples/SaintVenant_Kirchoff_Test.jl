using FlexiGal
Lx=5.0
Ly=1.0
Domain = (Lx, Ly)
Divisions = (150, 30)
dmax = 1.75
const E=1000.0;
const ν=0.3;
const G=E/(2*(1+ν));
const λ=E*ν/((1+ν)*(1-2*ν));
shift(x)=[x[1], x[2]-Ly/2];
const P=-1.2
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=shift)
ngpts = 3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
dΩ = IntegrationSet(Ω, ngpts)
dΓ2 = IntegrationSet(Γ2, ngpts) 
const u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
#const u₀=VectorField(0.0, 0.0)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2], VectorField{2,Float64}, dmax; Dirichlet_Boundaries=[Γ1],Dirichlet_Values=[u₀])
const s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))
const g=VectorField(0.0,-0.6)

NL_Problem = @NL_WeakForm u begin
    Egreen(u) = 1/2*(∇(u) + ∇(u)' + (∇(u)')*∇(u))
    dEgreen(u, du) = 1/2*(∇(du) + ∇(du)' + (∇(du)')*∇(u) + (∇(u)')*∇(du))
    Sₛ = 2*G*Egreen(u)
    Sᵥ = λ*tr(Egreen(u))*Id
    dSₛ(du) = 2*G*dEgreen(u, du)
    dSᵥ(du) = λ*tr(dEgreen(u, du))*Id
    @WeakForm Jac(δu, du) = ∫(∇(δu)⊙((Sₛ+Sᵥ)*∇(du)) + dEgreen(u,δu)⊙(dSₛ(du)+dSᵥ(du)))dΩ
    @WeakForm Res(δu)     = ∫(dEgreen(u,δu)⊙(Sₛ+Sᵥ))dΩ + ∫(-δu ⋅ s)dΓ2
end

op = NonLinearOperator(NL_Problem, Uspace);

uh = Prueba_Macro(op; tol = 1e-4, max_iter=50)