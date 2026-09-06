using FlexiGal
Lx =5.0 
Ly =1.0
Domain = (Lx, Ly)
nx=80; ny=30
Divisions = (nx, ny)
dmax = 1.5

function clustering(x)
    eps_val = 0.000001
    nx = Lx * (1.0 - tanh((1.0 - x[1] / Lx) * eps_val) / tanh(eps_val))
    ny = 0.5 + 0.5 * tanh((2.0 * x[2] - 1.0) * eps_val) / tanh(eps_val)
    return [nx, ny]
end

function dm_clustered(x0)
    eps_val = 0.000001
    s1 = eps_val / (tanh(eps_val) * cosh((1.0 - x0[1] / Lx) * eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
model = create_model(Domain, Divisions; map = clustering);
ngpts = 3
ng2 = 1
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Bottom")
Γ3 = Triangulation(model, "Top")
Γ4 = Triangulation(model, "Right")
dΩ = IntegrationSet(Ω, ngpts)
dΩₚ= IntegrationSet(Ω, ng2)
U₀(x) = VectorField(6*(x[2]-x[2]^2),0.0)
No_Slip = VectorField(0.0, 0.0)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3], VectorField{2,Float64}; dmax, dm_map = dm_clustered, method=:EFG, technique=:LME, shape=:rectangular, Dirichlet_Boundaries=[Γ1, Γ2, Γ3],
Dirichlet_Values=[VectorField(1.0, 0.0), No_Slip, No_Slip])
Pspace = ApproxSpace(model, [Ω], Float64; method=:FEM, order=0)
X = MultiApproxSpace(Uspace, Pspace)
@WeakForm a((δu, δp), (u, p)) = ∫(∇(δu) ⊙ (∇(u)+∇(u)'))dΩ - ∫(δp * (∇⋅u) + (∇⋅δu) * p )dΩ
@time op = Linear_Problem(a,nothing, X);
uh, ph = Solve(op, dΩ);
