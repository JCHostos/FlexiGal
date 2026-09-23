#This approxspace combination leads to checkerboard pattern in pressure
using FlexiGal
Lx = 1.0
Ly =1.0
Domain = (Lx, Ly)
nd=90
Divisions = (nd, nd)
dmax = 1.5
function clustering(x)
    eps_val=1.0
    nx = 0.5+ 0.5 * tanh((2.0*x[1]-1.0)*eps_val)/tanh(eps_val)
    ny = 0.5 + 0.5 * tanh((2.0*x[2]-1.0)*eps_val)/tanh(eps_val)
    return [nx, ny]
end
function dm_clustered(x0)
    eps_val=1.0
    s1 = eps_val / (tanh(eps_val) * cosh((2.0*x0[1]-1.0)*eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
model = create_model(Domain, Divisions; map=clustering);
ngpts = 3
ng2 = 3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Bottom")
Γ3 = Triangulation(model, "Right")
Γ4 = Triangulation(model, "Top")
dΩ = IntegrationSet(Ω, ngpts)
dΩₚ= IntegrationSet(Ω, ng2)
U₀(x) = VectorField((4*x[1]*(1-x[1]))^(1/8),0.0)
No_Slip = VectorField(0.0, 0.0)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3, Γ4], VectorField{2,Float64}; dmax, method=:EFG, technique=:LME, shape=:rectangular, Dirichlet_Entities=[Γ1, Γ2, Γ3, Γ4],
Dirichlet_Values=[No_Slip, No_Slip, No_Slip, VectorField(1.0,0.0)], dm_map=dm_clustered)
Pspace = ApproxSpace(model, [Ω], Float64; method=:FEM, order=0, constraint=:ZeroMean)
X = MultiApproxSpace(Uspace, Pspace)
@WeakForm a((δu, δp), (u, p)) = ∫(∇(δu) ⊙ (∇(u)+∇(u)') - (∇⋅δu) * p +  δp * (∇⋅u))dΩ 
@time op = Linear_Problem(a,nothing, X);
uh, ph = Solve(op, dΩ);
