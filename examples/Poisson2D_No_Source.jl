using FlexiGal
Domain = (1.0, 1.0)
Divisions = (80,80)
function clustering(x)
    eps_val=1.5
    nx = 0.5 + 0.5 * tanh((2.0*x[1]-1.0)*eps_val)/tanh(eps_val)
    ny = 0.5 + 0.5 * tanh((2.0*x[2]-1.0)*eps_val)/tanh(eps_val)
    return [nx, ny]
end

function dm_clustered(x0)
    eps_val=1.5
    s1 = eps_val / (tanh(eps_val) * cosh((2.0*x0[1]-1.0)*eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
dmax = 1.75
model = create_model(Domain, Divisions; map=clustering);
ngpts = 3
Ω = Triangulation(model, "Domain");
Γ1 = Triangulation(model, "Left");
Γ2 = Triangulation(model, "Bottom");
Γ3 = Triangulation(model, "Right");
Γ4 = Triangulation(model, "Top");
dΩ = IntegrationSet(Ω, ngpts);
Tspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3, Γ4],Float64; dmax, shape=:rectangular, method=:EFG, dm_map=dm_clustered, technique=:MK,
Dirichlet_Boundaries=[Γ1, Γ2, Γ3, Γ4], Dirichlet_Values=[0.0,x->5*x[1], 5.0, x->5*x[1]]);
const k = 1.0
const v(x) = VectorField(-100.0 * (x[2] - 0.5), 100.0 * (x[1] - 0.5))
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (k * ∇(T)) + δT * (v ⋅ ∇(T)))dΩ
@time op = Linear_Problem(a, Tspace);
@time Th = Solve(op,dΩ);