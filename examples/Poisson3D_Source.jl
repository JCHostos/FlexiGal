using FlexiGal
Domain = (1.0, 1.0, 1.0)
Divisions = (12, 12, 12)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = Triangulation(model, "Domain")
Γ = Triangulation(model, "Boundary")
dΩ = IntegrationSet(Ω,ngpts)
Tspace =  ApproxSpace(model, [Ω,Γ], Float64; dmax, Dirichlet_Entities=[Γ], Dirichlet_Values=[0.0], method=:EFG, technique=:IMLS);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
@WeakForm b(δT) = ∫(50 * δT)dΩ
@time op = Linear_Problem(a, b, Tspace)
@time Th = Solve(op,dΩ);