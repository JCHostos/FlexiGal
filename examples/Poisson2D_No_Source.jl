using FlexiGal
using Plots
Domain = (1.0, 1.0)
Divisions = (5, 5)
dmax = 1.65
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = BackgroundIntegration(model, "Domain", ngpts)
Γ1 = BackgroundIntegration(model, "Left", ngpts)
Γ2 = BackgroundIntegration(model, "Bottom", ngpts)
Measures = [Ω, Γ1, Γ2]
Tspace = EFGSpace(model, Measures, dm)
dΩ = EFG_Measure(Ω, Tspace)
Γd = [Γ1, Γ2]
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a)
dΓd = EFG_Measure(Γd, Tspace)
a(δT, T) = ∫(δT * (1000 * T)) * dΓd
Kp = Bilinear_Assembler(a)
Q = AssembleEFG(Γ2, Tspace, "Load"; prop=5000.0) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, Ω)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = Ω[2]
scatter(gs[:, 1], gs[:, 2], zcolor=Tgauss, color=:jet, marker=:square, markersize=1, markerstrokecolor=:transparent, markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature Colormap")