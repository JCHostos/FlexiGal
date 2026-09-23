using FlexiGal
using LinearAlgebra
include("/home/usuario/Documentos/Julia/FlexiGal/src/mmasub.jl")
include("/home/usuario/Documentos/Julia/FlexiGal/src/subsolv.jl")
Lx = 3.0
Ly = 1.0
Domain = (Lx, Ly)
Divisions = (240, 80)
dmax = 1.65
const E = 1.0;
const ν = 0.3;
const μₛ = E / (2 * (1 + ν));
const λₛ = E * ν / (1 - ν^2);
shift(x) = [x[1], x[2] - Ly / 2];
p = 3;
function Elastic_Moduli(λₛ, μₛ, p)
      λ(ρ) = λₛ * ρ^p + 1e-9 * λₛ * (1 - ρ^p)
      μ(ρ) = μₛ * ρ^p + 1e-9 * μₛ * (1 - ρ^p)
      dλdρ(ρ) = p * λₛ * ρ^(p - 1) - 1e-9 * p * λₛ * ρ^(p - 1)
      dμdρ(ρ) = p * μₛ * ρ^(p - 1) - 1e-9 * p * μₛ * ρ^(p - 1)
      return λ, μ, dλdρ, dμdρ
end
λ, μ, dλdρ, dμdρ = Elastic_Moduli(λₛ, μₛ, p)

P = -1.0
I₀ = Ly^3 / 12
model = create_model(Domain, Divisions)
ngpts = 2
Ω = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Right")
V2 = Triangulation(model,"Corner2")
dΓ2=IntegrationSet(Γ2,ngpts)
dΩ = IntegrationSet(Ω, ngpts)
#Displacement Space
Uspace = ApproxSpace(model, [Ω, Γ1, V2], VectorField{2,Float64}; method=:FEM,dmax=dmax, technique=:IMLS, Dirichlet_Entities=[Γ1,V2], Dirichlet_Values=[VectorField(0.0, 0.0),VectorField(0.0, 0.0)],Dirichlet_Masks = [[true, false],[false, true]])
#Adjoint Space (Always zero Dirichlet)
Λspace = ApproxSpace(model, [Ω, Γ1, V2], VectorField{2,Float64}; method=:FEM, dmax=dmax, technique=:IMLS, Dirichlet_Entities=[Γ1,V2], Dirichlet_Values=[VectorField(0.0, 0.0),VectorField(0.0, 0.0)],Dirichlet_Masks = [[true, false],[false, true]])
#Design Space 
Ρspace = ApproxSpace(model, [Ω], Float64; method=:EFG, dmax=1.35, technique=:LME, gamma=0.9)

ρ₀(x) = x[1] <= Lx / 2 ? 1.0 : 0.0

ρh0 = FlexiFunction(0.35, Ρspace, dΩ)
pt = PointSet([0.0, Ly])
Fc = VectorField(0.0, -P)
const s(x) = VectorField(0.0, P / (2 * I₀) * (Ly^2 / 4 - x[2]^2))
const g = VectorField(0.0, 0.0)
σ(u) = (μ ∘ ρh0) * (∇(u) + ∇(u)') + (λ ∘ ρh0) * (∇ ⋅ u) * Id
@WeakForm a(δu, u) = ∫(∇(δu) ⊙ σ(u))dΩ;
@WeakForm b(δu) =  δ(δu⋅Fc)pt;
@time op = Linear_Problem(a, b, Uspace);
uh = Solve(op, dΩ);
fobj00 = 0.5 * (∫(∇(uh) ⊙ σ(uh))dΩ)
σh=σ(uh)
@WeakForm aₛ(δu, γ) = ∫(∇(δu) ⊙ σ(γ))dΩ
@WeakForm dfobjdu(δu) = ∫(∇(δu) ⊙ σ(uh))dΩ
@time  opₛ = Linear_Problem(aₛ, dfobjdu, Λspace)

ρspace_built = build_space(Ρspace, [dΩ])

Flexi_params = (; op, opₛ, dΩ, dΓ2, Uspace, Λspace, Ρspace, ρspace_built, s, g, λ, μ)

function Phys_Solver(ρh, Flexi_params)
      dΩ = Flexi_params.dΩ
      dΓ2 = Flexi_params.dΓ2
      Uspace = Flexi_params.Uspace
      op = Flexi_params.op
      g = Flexi_params.g
      σ(u) = (μ ∘ ρh) * (∇(u) + ∇(u)') + (λ ∘ ρh) * (∇ ⋅ u) * Id
      @WeakForm a(δu, u) = ∫(∇(δu) ⊙ σ(u))dΩ
      @WeakForm b(δu) = δ(δu⋅Fc)pt;
      @time  op = Linear_Problem(a, b, Uspace, op)
      uh = Solve(op, dΩ);
      fobj = 0.5 * (∫(∇(uh) ⊙ σ(uh))dΩ)
      return uh, fobj
end

β=2.0;
η=0.5;

Threshold(x; β, η) = β == 0.0 ? x : (tanh(β * η) + tanh(β * (x - η))) / (tanh(β * η) + tanh(β * (1.0 - η)))
ρth = (ρₜ -> Threshold(ρₜ; β, η)) ∘ ρh0

dρth(x; β, η) = β == 0.0 ? 1.0 : β * (1.0 - tanh(β * (x - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))

function Sensitivity(ρh, β, η; Flexi_params)
      ρth = (ρₜ -> Threshold(ρₜ; β, η)) ∘ ρh
      uh, fobj = Phys_Solver(ρth, Flexi_params)
      dΩ = Flexi_params.dΩ
      opₛ = Flexi_params.opₛ
      Λspace = Flexi_params.Λspace
      ρspace_built = Flexi_params.ρspace_built
      Ρspace = Flexi_params.Ρspace
      dρth_dρ = (ρₜ -> dρth(ρₜ; β=β, η=η)) ∘ ρh
      σ(u) = (μ ∘ ρth) * (∇(u) + ∇(u)') + (λ ∘ ρth) * (∇ ⋅ u) * Id
      dσdρ(u) = (dμdρ ∘ ρth) * (∇(u) + ∇(u)') + (dλdρ ∘ ρth) * (∇ ⋅ u) * Id
      #σh = σ(uh)
      @WeakForm aₛ(δu, γ) = ∫(∇(δu) ⊙ σ(γ))dΩ
      @WeakForm dfobjdu(δu) = ∫(∇(δu) ⊙ σ(uh))dΩ
      println("Assembling adjoint problem...")
      @time opₛ = Linear_Problem(aₛ, dfobjdu, Λspace, opₛ)
      γh = Solve(opₛ, dΩ);
      #dσdρh = dσdρ(uh)
      @WeakForm dFobjdρ(δρ) = ∫((δρ *dρth_dρ) * ((0.5 *∇(uh)-∇(γh))⊙dσdρ(uh)))dΩ
      println("Assembling sensitivity vector...")
      @time dFobjdρ = Assemble_Vector(dFobjdρ,Ρspace, ρspace_built)
      return fobj, dFobjdρ, uh
end

function Optimizer(Flexi_params, model, ρh_init::FlexiFunction, β, volfrac; η=0.5, max_iter=200, tol=1e-3, plot_freq=20)
    dΩ = Flexi_params.dΩ
    Ρspace, ρspace_built = Flexi_params.Ρspace, Flexi_params.ρspace_built
    
    ρh = ρh_init
    xval = copy(Get_Nodal_Values(ρh))
    n = length(xval)
    
    xmin = zeros(Float64, n)
    xmax = ones(Float64, n)
    low = copy(xmin)
    upp = copy(xmax)
    xold1 = copy(xval)
    xold2 = copy(xval)
    
    Vol = ∫(1.0)dΩ  # o sum(∫(1.0)dΩ)
    
    m = 1
    a0 = 1.0
    a_mma = zeros(m)
    c = 1000.0 * ones(m)
    d = zeros(m)
    change = 1.0
    iter = 0
    history = Vector{Tuple{Int,Float64,Float64,Float64}}()

    while change > tol && iter < max_iter
        iter += 1
        
        # Campo proyectado (Threshold)
        ρth = (ρₜ -> Threshold(ρₜ; β=β, η=η)) ∘ ρh
        
        # 2. Problema físico y sensibilidades
        Fobj, df0dx, uh = Sensitivity(ρh, β, η; Flexi_params)
        
        # 3. Restricción de Volumen (MMA escalar)
        Volρth = ∫(ρth)dΩ
        println("Volume ", Volρth)
        max_vol = volfrac * Vol
        fval = Volρth / max_vol - 1.0
        
        dρth_dρ = (ρₜ -> dρth(ρₜ; β=β, η=η)) ∘ ρh
        @WeakForm dVol_form(δρ) = ∫(δρ * dρth_dρ)dΩ
        @time dVol_dρ = Assemble_Vector(dVol_form, Ρspace, ρspace_built)
        dfdx = dVol_dρ / max_vol

        # Paso MMA (sin reshape innecesario)
        xmma, = mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2,
                       df0dx/fobj00, fval, dfdx, low, upp, a0, a_mma, c, d)

        change = maximum(abs.(xmma .- xval))
        xold2 .= xold1
        xold1 .= xval
        xval .= xmma

        ρh = FlexiFunction(xval, ρh)

        push!(history, (iter, Fobj/fobj00, fval, change))
        #Fobj = Fobj / fobj00;
        println("It.: $iter | Change: $(round(change, digits=5)) | Fobj: $(round(Fobj, digits=4)) | Vol: $(round(fval, digits=4))")

        if iter % plot_freq == 0 || iter == 1
            fig = plot_field(model, ρth; title="Iteración $iter (β = $β)", scale=1.0)
            display(fig)
        end
    end

    return ρh, history
end

# Bucle de continuación
β_list = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0,64.0]
volfrac = 0.35
count = length(β_list)
History_list_β = Vector{Vector{Tuple{Int,Float64,Float64,Float64}}}(undef, count)

ρh_current = ρh0

for bi in 1:count
    global ρh_current, history
    β_val = β_list[bi]
    println("\n================ Running with β = $β_val ================")
    if bi ==1
    itermax=100
    else
    itermax=40
    end
    ρh_current, history = Optimizer(Flexi_params, model, ρh_current, β_val, volfrac; max_iter=itermax, tol=1e-3, plot_freq=5)
    History_list_β[bi] = history
end