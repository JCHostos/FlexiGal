using FlexiGal
Domain = (1.0, 1.0)
Divisions = (50,50)
dmax = 1.5
model = create_model(Domain, Divisions);
ngpts = 3;
Ω = Triangulation(model, "Domain");
Γ1 = Triangulation(model, "Left");
Γ2 = Triangulation(model, "Bottom");
Γ3 = Triangulation(model, "Right");
Γ4 = Triangulation(model, "Top");
dΩ = IntegrationSet(Ω, ngpts);
Tspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3, Γ4],Float64, dmax; 
Dirichlet_Boundaries=[Γ1, Γ2, Γ3, Γ4], Dirichlet_Values=[0.0,x->5*x[1], 5.0, x->5*x[1]]);
const k = 1.0
const v(x) = VectorField(-100.0 * (x[2] - 0.5), 100.0 * (x[1] - 0.5))
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (k * ∇(T)) + δT * (v ⋅ ∇(T)))dΩ
@time op = Linear_Problem(a, Tspace);
Th = Solve(op,dΩ);