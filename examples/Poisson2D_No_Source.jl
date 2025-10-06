using FlexiGal
using GLMakie
Domain = (1.0, 1.0)
Divisions = (150,150)
dmax = 1.05
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 2
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Bottom", ngpts)
dΓ3 = BackgroundIntegration(model, "Right", ngpts)
dΓ4 = BackgroundIntegration(model, "Top", ngpts)
@time Tspace = EFGSpace(model, [dΩ, dΓ1, dΓ2, dΓ3, dΓ4], dm)
k=1.0
w(x) = VectorField(600*(x[2]-0.5),-600*(x[1]-0.5))
a(δT, T) = ∫(∇(δT)⋅(k*∇(T))-δT*(w⋅∇(T)))dΩ
@time K = Bilinear_Assembler(a,Tspace)
dΓd=[dΓ1,dΓ2,dΓ3,dΓ4]
a(δT, T) = ∫(δT * (100000 * T))dΓd
Kp = Bilinear_Assembler(a,Tspace)
Q = AssembleEFG([dΓ3,dΓ4], Tspace, "Load"; prop=500000.0) # Non Null Dirichlet BC T=5
@time T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
fig = Figure()
ax = Axis(fig[1,1], aspect=Domain[2]/Domain[1])
scatter!(ax, gs[:,1], gs[:,2]; color=Tgauss, markersize=4, colormap=:jet)
fig