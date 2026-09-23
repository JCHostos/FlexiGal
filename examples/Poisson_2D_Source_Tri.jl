using FlexiGal
dmax = 1.5
include("Triangles_Mesh_Square.jl")
model = create_model(xy, conn)
ngpts = 1
Ω = Triangulation(model, "Domain");
Γ = Triangulation(model, "Boundary");
dΩ = IntegrationSet(Ω, ngpts);
Tspace =  ApproxSpace(model, [Ω, Γ],Float64; dmax, Dirichlet_Entities=[Γ], Dirichlet_Values=[0.0],shape=:circular, cell_based=true);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T))dΩ
@WeakForm b(δT) = ∫(50 * δT)dΩ
op = Linear_Problem(a, b, Tspace)
Th = Solve(op,dΩ);