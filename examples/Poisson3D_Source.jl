using FlexiGal
Domain = (1.0, 1.0, 1.0)
Divisions = (12, 12, 12)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ = BackgroundIntegration(model, "Boundary", ngpts)
@time Tspace = EFG_Space(model, [dΩ, dΓ],Float64, dm; Dirichlet_Measures=[dΓ])
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
b(δT) = ∫(5000 * δT)dΩ
@time A, F = Linear_Problem(a, b, Tspace)
@time T = A \ F;
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