using FlexiGal
using GLMakie
Domain = (1.0, 1.0, 0.1)
Divisions = (40, 40, 4)
dmax = 1.5
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Back", ngpts)
dΓd=[dΓ1,dΓ2]
Tspace = EFGSpace(model, [dΩ,dΓ1,dΓ2], dm)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a,Tspace)
a(δT, T) = ∫(δT * (1000.0 * T)) * dΓd
Kp = Bilinear_Assembler(a,Tspace)
b(δT)=∫(5000.0*δT)dΓ2
Q = Linear_Assembler(b,Tspace)
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
fig = Figure()
ax = Axis3(fig[1,1], aspect=Domain)
scatter!(ax, gs[:,1], gs[:,2], gs[:,3]; color=Tgauss, markersize=4, colormap=:jet)
fig