using FlexiGal
function clustering(x)
    eps_val=0.0000001
    nx = 0.5 + 0.5 * tanh((2.0*x[1]-1.0)*eps_val)/tanh(eps_val)
    ny = 0.5 + 0.5 * tanh((2.0*x[2]-1.0)*eps_val)/tanh(eps_val)
    return [nx, ny]
end

function dm_clustered(x0)
    eps_val=0.0000001
    s1 = eps_val / (tanh(eps_val) * cosh((2.0*x0[1]-1.0)*eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
dmax = 1.75
include("Triangles_Mesh_Square.jl")
model = create_model(xy, conn; map=clustering);
ngpts = 4
right(x) = x[1] > 1 - 1e-8
left(x) = x[1] < 1e-8
bottom(x) = x[2] < 1e-8
top(x) = x[2] > 1 - 1e-8
set_boundary_tag!(model, right; name="Right")
set_boundary_tag!(model, left; name="Left")
set_boundary_tag!(model, bottom; name="Bottom")
set_boundary_tag!(model, top; name="Top")
Ω = Triangulation(model, "Domain");
Γ1 = Triangulation(model, "Left");
Γ2 = Triangulation(model, "Bottom");
Γ3 = Triangulation(model, "Right");
Γ4 = Triangulation(model, "Top");
dΩ = IntegrationSet(Ω, ngpts);
Tspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3, Γ4],Float64; dmax, shape=:circular, cell_based=true, method=:EFG, dm_map=dm_clustered, technique=:IMLS,
Dirichlet_Entities=[Γ1, Γ2, Γ3, Γ4], Dirichlet_Values=[0.0,x->5*x[1], 5.0, x->5*x[1]]);
const k = 1.0
const v(x) = VectorField(-100.0 * (x[2] - 0.5), 100.0 * (x[1] - 0.5))
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (k * ∇(T)) + δT * (v ⋅ ∇(T)))dΩ
@time op = Linear_Problem(a, Tspace);
@time Th = Solve(op,dΩ);