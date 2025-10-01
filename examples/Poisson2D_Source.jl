using FlexiGal
using GLMakie
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.65
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ = BackgroundIntegration(model, "Boundary", ngpts)
Tspace = EFGSpace(model, [dΩ,dΓ], dm)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a,Tspace)
a(δT, T) = ∫(δT * (1000 * T)) * dΓ
Kp = Bilinear_Assembler(a,Tspace)
Q = AssembleEFG(dΩ, Tspace, "Load"; prop=5000) # Uniform Source
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
fig = Figure()
ax = Axis(fig[1,1], aspect=Domain[2]/Domain[1])
scatter!(ax, gs[:,1], gs[:,2]; color=Tgauss, markersize=4, colormap=:jet)
fig