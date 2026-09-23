using FlexiGal
Lx = 1.0
Ly =1.0
Domain = (Lx, Ly)
Divisions = (50, 50)
dmax = 2.0;
function clustering(x)
    eps_val=0.01
    nx = 0.5 + 0.5 * tanh((2.0*x[1]-1.0)*eps_val)/tanh(eps_val)
    ny = 0.5 + 0.5 * tanh((2.0*x[2]-1.0)*eps_val)/tanh(eps_val)
    return [nx, ny]
end
function dm_clustered(x0)
    eps_val=0.01
    s1 = eps_val / (tanh(eps_val) * cosh((2.0*x0[1]-1.0)*eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0*x0[2]-1.0)*eps_val)^2)
    return [s1, s2]
end
model = create_model(Domain, Divisions; map=clustering);
ngpts = 3
ngpts_red = 1
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Bottom")
Γ3 = Triangulation(model, "Right")
Γ4 = Triangulation(model, "Top")
dΩ = IntegrationSet(Ω, ngpts)
dΩᵣ = IntegrationSet(Ω, ngpts_red)
U₀(x) = VectorField(4*x[1]*(1-x[1]),0.0)
No_Slip = VectorField(0.0, 0.0)
Uspace = ApproxSpace(model, [Ω, Γ1, Γ2, Γ3, Γ4], VectorField{2,Float64}; dmax, method=:EFG, technique=:IMLS, shape=:rectangular, Dirichlet_Entities=[Γ1, Γ2, Γ3, Γ4],
Dirichlet_Values=[No_Slip, No_Slip, No_Slip, U₀], dm_map=dm_clustered)
    @WeakForm a(δu, u) = ∫(∇(δu)⊙(∇(u)+∇(u)'))dΩ + ∫((∇⋅δu) *100.0* (∇⋅u))dΩᵣ
@time op = Linear_Problem(a, Uspace);
uh = Solve(op);