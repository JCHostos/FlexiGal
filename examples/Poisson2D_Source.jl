using FlexiGal
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.5
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ = BackgroundIntegration(model, "Boundary", ngpts)
@time Tspace = EFG_Space(model, [dΩ, dΓ],Float64, dm; Dirichlet_Measures=[dΓ])
v(x) = VectorField(-150 * (x[2] - 0.5),150 * (x[1] - 0.5))
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)+δT * (v ⋅ ∇(T)))dΩ
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
    ax = Axis(fig[1, 1], aspect=Domain[2] / Domain[1])
    scatter!(ax, gs[:, 1], gs[:, 2]; color=Tgauss, markersize=4, colormap=:jet)
    fig
end