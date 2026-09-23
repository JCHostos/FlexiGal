using FlexiGal
using StaticArrays
Lx=4.0
Ly=1.0
Domain = (Lx, Ly)
div=10
Divisions = (4*div, div)
const E=250e9;
const ν=0.3;
const G=E/(2*(1+ν));
const λ = E*ν / (1 - ν^2);
shift(x)=[x[1], x[2]-Ly/2];
P=-10000000
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=shift)
ngpts = 3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
dΩ = IntegrationSet(Ω, ngpts)
dΓ2 = IntegrationSet(Γ2, ngpts) 
const u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2], VectorField{2,Float64}; method=:EFG,technique=:MK,dmax=2.0, shape=:rectangular, Dirichlet_Entities=[Γ1],Dirichlet_Values=[u₀])
const s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))
uexact(x) = VectorField(-P*x[2]/(6*E*I₀)*((6*Lx-3*x[1])*x[1]+(2+ν)*(x[2]^2-Ly^2/4)),P/(6*E*I₀)*(3*ν*x[2]^2*(Lx-x[1])+(4+5*ν)*Ly^2*x[1]/4+(3*Lx-x[1])*x[1]^2))
ε(u)=0.5*(∇(u)+∇(u)')
σ(ε)=2*G*ε+λ*tr(ε)*Id
@WeakForm a(δu, u) = ∫(∇(δu)⊙(G*(∇(u)+∇(u)'))+ (∇⋅δu)*λ*(∇⋅u))dΩ;
@WeakForm b(δu) = ∫(δu ⋅ s)dΓ2;
@time op = Linear_Problem(a, b, Uspace);
uh = Solve(op,dΩ);
num=∫((uh-uexact)⋅(uh-uexact))dΩ
uaux=uh;
reff=0.0*Get_Nodal_Values(uaux)
uaux=FlexiFunction(reff,uaux)
den=∫((uaux-uexact)⋅(uaux-uexact))dΩ
error1=sqrt(num/den)
εh=ε(uh)
σx(x) = -P*((Lx-x[1])*x[2])/I₀
εx(x) = σx(x)/E
εy(x) = -ν/E*σx(x)
τxy(x) = P/(2*I₀)*(Ly^2/4 - x[2]^2)
εxy(x) = τxy(x)/(2*G) 
ε_exact(x) = SMatrix{2,2,Float64}(εx(x), εxy(x), εxy(x), εy(x))
num=∫((εh-ε_exact)⊙σ(εh-ε_exact))dΩ
εhaux=ε(uaux)
den = ∫((εhaux-ε_exact)⊙σ(εhaux-ε_exact))dΩ
error2=sqrt(num/den)