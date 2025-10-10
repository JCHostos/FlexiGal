using FlexiGal
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Bottom", ngpts)
dΓ3 = BackgroundIntegration(model, "Right", ngpts)
dΓ4 = BackgroundIntegration(model, "Top", ngpts)
@time Tspace = EFG_Space(model, [dΩ, dΓ1, dΓ2, dΓ3, dΓ4],Float64, dm; 
Dirichlet_Measures=[dΓ1, dΓ2, dΓ3, dΓ4], Dirichlet_Values=[0.0,x->5*x[1], 5.0, x->5*x[1]])
k = 1.0
v(x) = VectorField(150.0,0.0)
#w = VectorField(0.0,0.0)
a(δT, T) = ∫(∇(δT) ⋅ (k * ∇(T)) + δT * (v ⋅ ∇(T)))dΩ
@time A, F = Linear_Problem(a, Tspace)
@time T = A \ F;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
∇Tgauss = Get_Point_Values(∇Th)
gs = dΩ.gs
if !haskey(ENV, "GITHUB_ACTIONS")
    using GLMakie
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=Domain[2] / Domain[1])
    scatter!(ax, gs[:, 1], gs[:, 2]; color=Tgauss, markersize=4, colormap=:jet)
    fig
end