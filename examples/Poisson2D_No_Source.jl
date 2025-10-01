using FlexiGal
using Plots
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.65
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Bottom", ngpts)
dΓd, model = Merge_Measures(model,[dΓ1,dΓ2], tag="Dirichlet")
Tspace = EFGSpace(model, [dΩ, dΓd, dΓ2], dm)
a(δT, T) = ∫(∇(δT) ⋅ ∇(T)) * dΩ
K = Bilinear_Assembler(a,Tspace)
a(δT, T) = ∫(δT * (1000 * T)) * dΓd
Kp = Bilinear_Assembler(a,Tspace)
Q = AssembleEFG(dΓ2, Tspace, "Load"; prop=5000.0) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
Th = EFGFunction(T, Tspace, dΩ)
Tgauss = Get_Point_Values(Th)
∇Th = ∇(Th)
gs = dΩ.gs
scatter(gs[:, 1], gs[:, 2], zcolor=Tgauss, color=:jet, marker=:square, markersize=1, markerstrokecolor=:transparent, markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature Colormap")