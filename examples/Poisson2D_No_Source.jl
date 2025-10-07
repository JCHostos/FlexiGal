using FlexiGal
using GLMakie
Domain = (1.0, 1.0)
Divisions = (100,100)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Bottom", ngpts)
dΓ3 = BackgroundIntegration(model, "Right", ngpts)
dΓ4 = BackgroundIntegration(model, "Top", ngpts)
Dirichlet_Measures=[dΓ1, dΓ2, dΓ3, dΓ4]
Dirichlet_Values=[0.0,0.0,5.0,5.0]
@time Tspace = EFG_Space(model, [dΩ, dΓ1, dΓ2, dΓ3, dΓ4], dm, Dirichlet_Measures, Dirichlet_Values)
k=1.0
w(x) = VectorField(150*(x[2]-0.5),-150*(x[1]-0.5))
a(δT, T) = ∫(∇(δT)⋅(k*∇(T))-δT*(w⋅∇(T)))dΩ
@time A,F=Linear_Problem(a,Tspace)
@time T = A\F;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
fig = Figure()
ax = Axis(fig[1,1], aspect=Domain[2]/Domain[1])
scatter!(ax, gs[:,1], gs[:,2]; color=Tgauss, markersize=4, colormap=:jet)
fig