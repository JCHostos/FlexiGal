using FlexiGal
using GLMakie
Domain = (1.0, 1.0, 0.1)
Divisions = (50, 50, 5)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Back", ngpts)
Tspace = EFGSpace(model, [dΩ,dΓ1,dΓ2], dm)
dΩ = EFG_Measure(Ω, Tspace)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a)
dΓ = EFG_Measure([Γ1,Γ2], Tspace)
a(δT, T) = ∫(δT * (1000.0 * T)) * dΓ
Kp = Bilinear_Assembler(a)
Q = AssembleEFG(Γ2, Tspace, "Load"; prop=5000.0) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, Ω)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = Ω[2]
fig = Figure()
ax = Axis3(fig[1,1], aspect=Domain)
scatter!(ax, gs[:,1], gs[:,2], gs[:,3]; color=Tgauss, markersize=4, colormap=:jet)
fig