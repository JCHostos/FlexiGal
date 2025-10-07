using FlexiGal
Domain = (1.0, 1.0, 0.1)
Divisions = (40, 40, 4)
dmax = [1.35, 1.35, 1.05]
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Back", ngpts)
dΓd = [dΓ1, dΓ2]
Dirichlet_Measures = [dΓ1, dΓ2]
Dirichlet_Values = [0.0, 5.0]
@time Tspace = EFG_Space(model, [dΩ, dΓ1, dΓ2], dm, Dirichlet_Measures, Dirichlet_Values)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
@time A, F = Linear_Problem(a, Tspace)
@time T = A \ F
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
if !haskey(ENV, "GITHUB_ACTIONS")
    using GLMakie
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=Domain)
    scatter!(ax, gs[:, 1], gs[:, 2], gs[:, 3]; color=Tgauss, markersize=4, colormap=:jet)
    fig
end