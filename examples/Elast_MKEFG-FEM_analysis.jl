using FlexiGal
using StaticArrays
Lx=4.0
Ly=1.0
Domain = (Lx, Ly)
div=10
Divisions = (4*div, div)
dmax = 2.0
const E=250e9;
const ν=0.3;
const G=E/(2*(1+ν));
const λ = E*ν / (1 - ν^2);
shift(x)=[x[1], x[2]-Ly/2];
P=-10000000
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=shift)
ngpts = 3
MK(x)=x[1]<1.0
FEM(x) = !MK(x)
set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
Ω_fem = Triangulation(model, "FEM")
Ω_mk = Triangulation(model, "MK")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
dΩ_fem = IntegrationSet(Ω_fem, ngpts-1)
dΩ_mk = IntegrationSet(Ω_mk, ngpts)
dΓ2 = IntegrationSet(Γ2, ngpts) 
const u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
Uspace = ApproxSpace(model, [Ω_mk,Ω_fem, Γ1, Γ2], VectorField{2,Float64}; dmax=dmax, method=[:EFG,:FEM,:EFG,:FEM], Cq=4.3, technique=:MK, shape=:rectangular, Dirichlet_Entities=[Γ1],Dirichlet_Values=[u₀])
const s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))
uexact(x) = VectorField(-P*x[2]/(6*E*I₀)*((6*Lx-3*x[1])*x[1]+(2+ν)*(x[2]^2-Ly^2/4)),P/(6*E*I₀)*(3*ν*x[2]^2*(Lx-x[1])+(4+5*ν)*Ly^2*x[1]/4+(3*Lx-x[1])*x[1]^2))
const g=VectorField(0.0,-1.0)
ε(u)=0.5*(∇(u)+∇(u)')
σ(ε)=2*G*ε+λ*tr(ε)*Id
@WeakForm a(δu, u) = ∫(∇(δu)⊙(G*(∇(u)+∇(u)'))+ (∇⋅δu)*λ*(∇⋅u))dΩ_mk + ∫(∇(δu)⊙(G*(∇(u)+∇(u)'))+ (∇⋅δu)*λ*(∇⋅u))dΩ_fem;
@WeakForm b(δu) = ∫(δu ⋅ s)dΓ2;
@time op = Linear_Problem(a, b, Uspace);
uh = Solve(op);
num=∫((uh[dΩ_fem]-uexact)⋅(uh[dΩ_fem]-uexact))dΩ_fem + ∫((uh[dΩ_mk]-uexact)⋅(uh[dΩ_mk]-uexact))dΩ_mk
uaux=copy(uh);
uauxfem=uaux[dΩ_fem]
uauxmk=uaux[dΩ_mk]
reff=0.0*Get_Nodal_Values(uauxfem)
uauxfem=FlexiFunction(reff,uauxfem)
uauxmk=FlexiFunction(reff,uauxmk)
den=∫((uauxfem-uexact)⋅(uauxfem-uexact))dΩ_fem + ∫((uauxmk-uexact)⋅(uauxmk-uexact))dΩ_mk
error1=sqrt(num/den)

εh_fem=ε(uh[dΩ_fem])
εh_mk=ε(uh[dΩ_mk])
σx(x) = -P*((Lx-x[1])*x[2])/I₀
εx(x) = σx(x)/E
εy(x) = -ν/E*σx(x)
τxy(x) = P/(2*I₀)*(Ly^2/4 - x[2]^2)
εxy(x) = τxy(x)/(2*G) 
ε_exact(x) = SMatrix{2,2,Float64}(εx(x), εxy(x), εxy(x), εy(x))
num=∫((εh_fem-ε_exact)⊙σ(εh_fem-ε_exact))dΩ_fem + ∫((εh_mk-ε_exact)⊙σ(εh_mk-ε_exact))dΩ_mk
εhaux_fem=ε(uauxfem)
εhaux_mk=ε(uauxmk)
den = ∫((εhaux_fem-ε_exact)⊙σ(εhaux_fem-ε_exact))dΩ_fem + ∫((εhaux_mk-ε_exact)⊙σ(εhaux_mk-ε_exact))dΩ_mk
error2=sqrt(num/den)