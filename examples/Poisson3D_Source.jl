using FlexiGal
using Plots
Domain = (1.0, 1.0, 1.0)
Divisions = (20, 20, 20)
dmax = 1.35
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = BackgroundIntegration(model, "Domain", ngpts)
Γ = BackgroundIntegration(model, "Boundary", ngpts)
Measures = [Ω, Γ]
Tspace = EFGSpace(model, Measures, dm)
dΩ = EFG_Measure(Ω, Tspace)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a)
dΓ = EFG_Measure(Γ, Tspace)
a(δT, T) = ∫(δT * (1000 * T)) * dΓ
Kp = Bilinear_Assembler(a)
Q = AssembleEFG(Ω, Tspace, "Load"; prop=5000) # Uniform Source
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, Ω)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = Ω[2]
fig = Figure()
ax = Axis3(fig[1,1], aspect=Domain)
scatter!(ax, gs[:,1], gs[:,2], gs[:,3]; color=Tgauss, markersize=4, colormap=:jet)
fig