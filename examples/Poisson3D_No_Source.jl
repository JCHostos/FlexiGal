using FlexiGal
Domain = (1.0, 1.0, 0.1)
Divisions = (40, 40, 4)
dmax = [1.5, 1.5, 1.05]
model = create_model(Domain, Divisions)
ngpts =3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Back")
dΩ = IntegrationSet(Ω, ngpts)
Tspace =  ApproxSpace(model, [Ω,Γ1,Γ2], Float64; dmax, Dirichlet_Boundaries=[Γ1,Γ2],Dirichlet_Values=[0.0,5.0]);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
@time op = Linear_Problem(a, Tspace);
Th = Solve(op,dΩ);