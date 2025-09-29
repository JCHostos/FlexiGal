using FlexiGal
using Plots
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.5
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = BackgroundIntegration(model, "Domain", ngpts)
Γ1 = BackgroundIntegration(model, "Left", ngpts)
Γ2 = BackgroundIntegration(model, "Bottom", ngpts)
Measures = [Ω, Γ1, Γ2]
Tspace = EFGSpace(model, Measures, dm)
K = AssembleEFG(Ω, Tspace, "Laplacian"; prop=1)
Γd = [Γ1, Γ2]
Kp = AssembleEFG(Γd, Tspace, "Mass"; prop=1000)
Q = AssembleEFG(Γ2, Tspace, "Load"; prop=5000) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
#Cálculo de Campo en puntos de Gauss (Pronto una función para esto en cualquier Tag donde haya Shape_Functions calculadas)
Th=EFGFunction(T, Tspace, Ω)
Tgauss=Get_Point_Values(Th)
∇Th=∇(Th)
gs=Ω[2]
scatter(gs[:,1], gs[:,2], zcolor=Tgauss, color=:jet, marker=:square, markersize=1,markerstrokecolor=:transparent,markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature Colormap")