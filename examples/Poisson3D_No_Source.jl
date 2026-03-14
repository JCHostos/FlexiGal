using FlexiGal
Domain = (1.0, 1.0, 0.1)
Divisions = (29, 29, 2)
dmax = [1.35, 1.35, 1.05]
model = create_model(Domain, Divisions)
ngpts =3
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Back")
dΩ = IntegrationSet(Ω, ngpts)
Tspace =  ApproxSpace(model, [Ω,Γ1,Γ2], Float64, dmax; Dirichlet_Boundaries=[Γ1,Γ2],Dirichlet_Values=[0.0,5.0]);
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
@time op = Linear_Problem(a, Tspace);
Th = Solve(op,dΩ); 
#=using GLMakie
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=Domain)
    scatter!(ax, gs[:, 1], gs[:, 2], gs[:, 3]; color=Tgauss, markersize=4, colormap=:jet)
fig=#