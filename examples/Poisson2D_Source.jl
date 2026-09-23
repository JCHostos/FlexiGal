using FlexiGal
Domain = (1.0, 1.0)
Divisions = (50, 50)
dmax = 1.5
model = create_model(Domain, Divisions)
ngpts = 3
Ω = Triangulation(model, "Domain");
Γ = Triangulation(model, "Boundary");
dΩ = IntegrationSet(Ω, ngpts);
Tspace =  ApproxSpace(model, [Ω, Γ],Float64; dmax, method=:EFG,technique=:IMLS, Dirichlet_Entities=[Γ], Dirichlet_Values=[0.0],shape=:circular);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T))dΩ
@WeakForm b(δT) = ∫(50 * δT)dΩ
@time op = Linear_Problem(a, b, Tspace)
Th = Solve(op,dΩ);