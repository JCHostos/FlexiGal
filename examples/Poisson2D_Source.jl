using FlexiGal
using Plots
Domain = (1.0, 1.0)
Divisions = (100, 100)
dmax = 1.5
model = create_model(Domain, Divisions)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
Ω = BackgroundIntegration(model, "Domain", ngpts)
Γ = BackgroundIntegration(model, "Boundary", ngpts)
Measures = [Ω, Γ]
Shape_Functions = EFGSpace(model, Measures, dm)
K = AssembleEFG(model, Ω, Shape_Functions, "Laplacian"; prop=1)
Kp = AssembleEFG(model, Γ, Shape_Functions, "Mass"; prop=1000)
Q = AssembleEFG(model, Ω, Shape_Functions, "Load"; prop=5000) # Uniform Source
T = (K + Kp) \ Q;
Th=EFGFunction(T, Shape_Functions, Ω)
Tgauss=Get_Point_Values(Th)
∇Th=∇(Th)
gs=Ω[2]
scatter(gs[:,1], gs[:,2],Tgauss, color=:blue, marker=:square, markersize=1,markerstrokecolor=:transparent,markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature")