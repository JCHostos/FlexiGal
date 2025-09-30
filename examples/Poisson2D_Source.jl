using FlexiGal
using Plots
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.65
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = BackgroundIntegration(model, "Domain", ngpts)
Γ = BackgroundIntegration(model, "Boundary", ngpts)
Measures = [Ω, Γ]
Tspace = EFGSpace(model, Measures, dm)
dΩ = EFG_Measure(Ω, Tspace)
a(δT, T, dΩ) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a, dΩ)
dΓ = EFG_Measure(Γ, Tspace)
a(δT, T, dΓ) = ∫(δT * (1000 * T)) * dΓ
Kp = Bilinear_Assembler(a, dΓ)
Q = AssembleEFG(Ω, Tspace, "Load"; prop=5000) # Uniform Source
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, Ω)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = Ω[2]
scatter(gs[:, 1], gs[:, 2], Tgauss, color=:blue, marker=:square, markersize=1, markerstrokecolor=:transparent, markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature")