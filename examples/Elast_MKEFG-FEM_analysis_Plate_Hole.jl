using FlexiGal
using StaticArrays
using LinearAlgebra
include("Plate_Hole.jl");
model = create_model(xy, conn);
dmax = 2.0
const E=210e9;
const ν=0.3;
const G=E/(2*(1+ν));
const λ = E*ν / (1-ν^2);
T=1000
ngpts = 3
MK(x)=x[1]<2.0 && x[2]<2.0
FEM(x) = !MK(x)
Bottom(x) = x[2] < 0.00001;
Left(x) = x[1] < 0.00001
Right(x) = x[1] > 10.0 - 0.00001;
Top(x) = x[2] > 10.0 - 0.00001
set_boundary_tag!(model, Bottom; name="Bottom"); set_boundary_tag!(model, Left; name="Left");
set_boundary_tag!(model, Top; name="Top"); set_boundary_tag!(model, Right; name="Right"); 
set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
Ω_fem = Triangulation(model, "FEM")
Ω_mk = Triangulation(model, "MK")
Γₗ = Triangulation(model, "Left")
Γᵣ = Triangulation(model, "Right")
Γᵦ = Triangulation(model, "Bottom")
Γₜ = Triangulation(model, "Top")
dΩ_fem = IntegrationSet(Ω_fem, ngpts-1)
dΩ_mk = IntegrationSet(Ω_mk, ngpts)
dΓₜ = IntegrationSet(Γₜ, ngpts)
dΓᵣ = IntegrationSet(Γᵣ, ngpts)
Uspace = ApproxSpace(model, [Ω_mk,Ω_fem, Γₗ, Γᵦ, Γᵣ, Γₜ], VectorField{2,Float64}; dmax=dmax, method=[:EFG,:FEM,:FEM,:FEM,:FEM,:FEM], technique=:MK, Cq=4.3, shape=:circular, cell_based = true, Dirichlet_Entities=[Γₗ,Γᵦ],Dirichlet_Values=[VectorField(0.0,0.0),VectorField(0.0,0.0)],Dirichlet_Masks=[[true,false],[false,true]])
R = 1.0 # Radio del agujero (1.0 m)
κ = (3 - ν) / (1 + ν)
r(x) = sqrt(x[1]^2 + x[2]^2)
θ(x) = atan(x[2], x[1]) 
σx(x)  =  T * (1 - (R/r(x))^2 * (3/2 * cos(2*θ(x)) + cos(4*θ(x))) + 3/2 * (R/r(x))^4 * cos(4*θ(x)))
σy(x)  = -T * ((R/r(x))^2 * (1/2 * cos(2*θ(x)) - cos(4*θ(x))) + 3/2 * (R/r(x))^4 * cos(4*θ(x)))
τxy(x) = -T * ((R/r(x))^2 * (1/2 * sin(2*θ(x)) + sin(4*θ(x))) - 3/2 * (R/r(x))^4 * sin(4*θ(x)))
ur(x) = (T * r(x) / (8 * G)) * ( (κ - 1) + 2 * (R/r(x))^2 + (2 + 2*(κ + 1)*(R/r(x))^2 - 2*(R/r(x))^4) * cos(2*θ(x)) )
uθ(x) = -(T * r(x) / (8 * G)) * ( 2 * (1 + (κ - 1)*(R/r(x))^2 + (R/r(x))^4) * sin(2*θ(x)) )
ux(x) = ur(x) * cos(θ(x)) - uθ(x) * sin(θ(x))
uy(x) = ur(x) * sin(θ(x)) + uθ(x) * cos(θ(x))
const sₜ(x)=VectorField(τxy(x),σy(x))
const sᵣ(x)=VectorField(σx(x),τxy(x))
ε(u)=0.5*(∇(u)+∇(u)')
σ(ε)=2*G*ε+λ*tr(ε)*Id
@WeakForm a(δu, u) = ∫(∇(δu)⊙(G*(∇(u)+∇(u)'))+ (∇⋅δu)*λ*(∇⋅u))dΩ_mk + ∫(∇(δu)⊙(G*(∇(u)+∇(u)'))+ (∇⋅δu)*λ*(∇⋅u))dΩ_fem;
@WeakForm b(δu) = ∫(δu ⋅ sₜ)dΓₜ + ∫(δu ⋅ sᵣ)dΓᵣ;
@time op = Linear_Problem(a, b, Uspace);
uh = Solve(op);
uexact(x)=VectorField(ux(x),uy(x));
num=∫((uh[dΩ_fem]-uexact)⋅(uh[dΩ_fem]-uexact))dΩ_fem + ∫((uh[dΩ_mk]-uexact)⋅(uh[dΩ_mk]-uexact))dΩ_mk
uaux=copy(uh);
uauxfem=uaux[dΩ_fem]
uauxmk=uaux[dΩ_mk]
reff=0.0*Get_Nodal_Values(uauxfem)
uauxfem=FlexiFunction(reff,uauxfem)
uauxmk=FlexiFunction(reff,uauxmk)
den=∫((uauxfem-uexact)⋅(uauxfem-uexact))dΩ_fem + ∫((uauxmk-uexact)⋅(uauxmk-uexact))dΩ_mk
error1=sqrt(num/den)
#=
εh_fem=ε(uh[dΩ_fem])
εh_mk=ε(uh[dΩ_mk])
εx(x) = σx(x)/E - -ν/E*σy(x)
εy(x) = σy(x)/E -ν/E*σx(x)
εxy(x) = τxy(x)/(2*G) 
ε_exact(x) = SMatrix{2,2,Float64}(εx(x), εxy(x), εxy(x), εy(x))
num=∫((εh_fem-ε_exact)⊙σ(εh_fem-ε_exact))dΩ_fem + ∫((εh_mk-ε_exact)⊙σ(εh_mk-ε_exact))dΩ_mk
εhaux_fem=ε(uauxfem)
εhaux_mk=ε(uauxmk)
den = ∫((εhaux_fem-ε_exact)⊙σ(εhaux_fem-ε_exact))dΩ_fem + ∫((εhaux_mk-ε_exact)⊙σ(εhaux_mk-ε_exact))dΩ_mk
error2=sqrt(num/den)
=#