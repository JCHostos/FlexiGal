using FlexiGal
using LinearAlgebra
include("/home/hostos/Documents/FlexiGal/examples/Triangles_Ring.jl");
dmax = 1.75;
ngpts = 4;
model = create_model(xy, conn);
const Tinner = 900; const Touter = 940.0;
const kₛ = 231.0; const kₗ = 90.7; const cpₛ = 900.0; const cpₗ = 1300.0;
const ρ = 2700.0; const Hf = 396400.0; const Tₗ = 930; const Tₛ = 919;

fₛ(T) = T <= Tₛ ? 1.0 : (T >= Tₗ ? 0.0 : 1.0 - (T - Tₛ) / (Tₗ - Tₛ))
inner(x) = x[1]^2 + x[2]^2 < 1.0^2 + 1e-1; outer(x) = x[1]^2 + x[2]^2 >= 2.0^2 - 1e-1;
Ω = Triangulation(model, "Domain");
set_boundary_tag!(model, inner; name="inner"); set_boundary_tag!(model, outer; name="outer");
Γinner = Triangulation(model, "inner"); Γouter = Triangulation(model, "outer");
dΩ = IntegrationSet(Ω, ngpts);

Tspace = ApproxSpace(model, [Ω, Γinner, Γouter], Float64; dmax, method=:FEM, technique =:MK,shape=:circular, 
cell_based=true, Dirichlet_Boundaries=[Γinner,Γouter], Dirichlet_Values=[Tinner,x->Touter + 100* sin(atan(x[2], x[1]))]);

const v(x) = VectorField(-1e-3*x[2]/(x[1]^2 + x[2]^2)*(1.0-exp(-(x[1]^2 + x[2]^2))), 1e-3*x[1]/(x[1]^2 + x[2]^2)*(1.0-exp(-(x[1]^2 + x[2]^2))))
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρ * cpₛ * v ⋅ ∇(T)))dΩ
op = Linear_Problem(a,Tspace);
Th=Solve(op)