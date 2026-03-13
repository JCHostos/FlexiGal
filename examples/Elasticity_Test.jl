using FlexiGal
Lx=5.0
Ly=1.0
Domain = (Lx, Ly)
Divisions = (50, 10)
dmax = 1.75
const E=1000.0;
const ν=0.3;
const G=E/(2*(1+ν));
const λ=E*ν/((1+ν)*(1-2*ν));
shift(x)=[x[1], x[2]-Ly/2];
P=-0.001
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=shift)
ngpts = 3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
dΩ = IntegrationSet(Ω, ngpts)
dΓ2 = IntegrationSet(Γ2, ngpts) 
const u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2], VectorField{2,Float64}, dmax; Dirichlet_Boundaries=[Γ1],Dirichlet_Values=[u₀])
const s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))
const g=VectorField(0.0,-1.0)
σ(u)=G*(∇(u)+∇(u)')+λ*(∇⋅u)*Id
@WeakForm a(δu, u) = ∫(∇(δu)⊙σ(u))dΩ;
@WeakForm b(δu) = ∫(δu ⋅ s)dΓ2;
op = Linear_Problem(a, b, Uspace);
uh = Solve(op);

#=op2 = let uh=uh
Egreen(u)=1/2*(∇(u)+∇(u)'+(∇(u)')*∇(u))
dEgreen(du)=1/2*(∇(du)+∇(du)'+(∇(du)')*∇(uh)+(∇(uh)')*∇(du))
Sₛ = 2*G*Egreen(uh)
Sᵥ = λ*tr(Egreen(uh))*Id
dSₛ(du) = 2*G*dEgreen(du)
dSᵥ(du) = λ*tr(dEgreen(du))*Id
@WeakForm a2(δu, du) = ∫(∇(δu)⊙((Sₛ+Sᵥ)*∇(du))+dEgreen(δu)⊙(dSₛ(du)+dSᵥ(du)))dΩ
@WeakForm b(δu)  = ∫((dEgreen(δu))⊙(Sₛ+Sᵥ))dΩ - ∫(δu ⋅ s)dΓ2
@time Linear_Problem(a2, b, Uspace);
end;
uh2 = Solve(op2,dΩ);=#