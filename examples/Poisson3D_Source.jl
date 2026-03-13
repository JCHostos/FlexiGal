using FlexiGal
Domain = (1.0, 1.0, 1.0)
Divisions = (12, 12, 12)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = Triangulation(model, "Domain")
Γ = Triangulation(model, "Boundary")
dΩ = IntegrationSet(Ω,ngpts)
Tspace =  ApproxSpace(model, [Ω,Γ], Float64, dmax; Dirichlet_Boundaries=[Γ]);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
@WeakForm b(δT) = ∫(5000 * δT)dΩ
@time op = Linear_Problem(a, b, Tspace)
@time Th = Solve(op,dΩ)
Tgauss = Get_Point_Values(Th)
gs = Th.Measure.gs
#=if !haskey(ENV, "GITHUB_ACTIONS")
    using GLMakie
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=Domain)
    scatter!(ax, gs[:, 1], gs[:, 2], gs[:, 3]; color=Tgauss, markersize=4, colormap=:jet)
    fig
end=#