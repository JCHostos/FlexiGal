using FlexiGal
Lx=5.0
Ly=1.0
Domain = (Lx, Ly)
Divisions = (200, 40)
dmax = 1.75
E=1000.0;
ν=0.3;
G=E/(2*(1+ν));
λ=E*ν/((1+ν)*(1-2*ν));
shift(x)=[x[1], x[2]-0.5];
P=-1.0
I₀=Ly^3/12
model = create_model(Domain, Divisions; map=shift)
dm = Influence_Domains(model, Domain, Divisions, dmax)
ngpts = 3
dΩ = BackgroundIntegration(model, "Domain", ngpts)
dΓ1 = BackgroundIntegration(model, "Left", ngpts)
dΓ2 = BackgroundIntegration(model, "Right", ngpts)
u₀(x)=VectorField(-P*(2+ν)*x[2]/(6*E*I₀)*(x[2]^2-Ly^2/4), P*ν*Lx/(2*E*I₀)*x[2]^2)
@time Uspace = EFG_Space(model, [dΩ, dΓ1, dΓ2], VectorField{2,Float64}, dm; Dirichlet_Measures=[dΓ1],Dirichlet_Values=[u₀])
s(x)=VectorField(0.0,P/(2*I₀)*(Ly^2/4 - x[2]^2))
# Bilinear y linear
a(δu, u) = ∫(G*(∇(δu) ⊙ (∇(u)+∇(u)')) + λ*((∇⋅δu)*(∇⋅u)))dΩ
b(δu)    = ∫(δu ⋅ s)dΓ2

@time A, F = Linear_Problem(a, b, Uspace)
@time u = A \ F

uh = EFGFunction(u, Uspace, dΩ)
ugauss = Get_Point_Values(uh)

# 1. Extraemos ux y uy de ugauss
ux = [v[1] for v in ugauss]
uy = [v[2] for v in ugauss]
mod_u = [sqrt(v[1]^2 + v[2]^2) for v in ugauss]
σh = G * (∇(uh) + ∇(uh)') + λ * (∇ ⋅ uh)
σh_gauss = Get_Point_Values(σh)
σₓₓ = [v[1,1] for v in σh_gauss]
# 2. Definimos un factor de amplificación (ajustalo si no se ve nada)
factor = 1.0 

gs = dΩ.gs
if !haskey(ENV, "GITHUB_ACTIONS")
    using GLMakie
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=Domain[1] / Domain[2])
    scatter!(ax, gs[:, 1] .+ ux .* factor, gs[:, 2] .+ uy .* factor; 
             color=σₓₓ, markersize=4, colormap=:jet)
    display(fig)
end