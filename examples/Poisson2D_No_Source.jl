using FlexiGal
using GLMakie
Domain = (1.0, 1.0)
Divisions = (50,50)
dmax = 1.5
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Bottom", ngpts)
Tspace = EFGSpace(model, [dΩ, dΓ1, dΓ2], dm)
k(x)=1
a(δT, T) = ∫(∇(δT) ⋅ (k*∇(T))) * dΩ
K = Bilinear_Assembler(a,Tspace)
dΓd=[dΓ1,dΓ2]
a(δT, T) = ∫(δT * (1000 * T)) * dΓd
Kp = Bilinear_Assembler(a,Tspace)
Q = AssembleEFG(dΓ2, Tspace, "Load"; prop=5000.0) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
fig = Figure()
ax = Axis(fig[1,1], aspect=Domain[2]/Domain[1])
scatter!(ax, gs[:,1], gs[:,2]; color=Tgauss, markersize=4, colormap=:jet)
fig