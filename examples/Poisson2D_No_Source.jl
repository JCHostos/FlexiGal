using FlexiGal
using LinearAlgebra
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
Shape_Functions = EFG_Functions(model, Measures, dm)
K = AssembleEFG(model, Ω, Shape_Functions, "Laplacian"; prop=1)
Γd = [Γ1, Γ2]
Kp = AssembleEFG(model, Γd, Shape_Functions, "Mass"; prop=1000)
Q = AssembleEFG(model, Γ2, Shape_Functions, "Load"; prop=5000) # Non Null Dirichlet BC T=5
T = (K + Kp) \ Q;
#Cálculo de Campo en puntos de Gauss (Pronto una función para esto en cualquier Tag donde haya Shape_Functions calculadas)
ngauss = size(Ω[2], 1)
PHI, DPHI, DOM = Shape_Functions[:domain]["Domain"]
Tdom = Vector{Vector{Float64}}(undef, ngauss)
Tgauss = Vector{Float64}(undef, ngauss)
@inbounds for i in 1:ngauss
    Tdom[i] = T[DOM[i]]
end
@inbounds for i in 1:ngauss
    Tgauss[i] = dot(PHI[i], Tdom[i])
end
gs = Ω[2]
scatter(gs[:,1], gs[:,2], zcolor=Tgauss, color=:jet, marker=:square, markersize=1,markerstrokecolor=:transparent,markerstrokewidth=0, xlabel="X", ylabel="Y", title="Temperature Colormap")