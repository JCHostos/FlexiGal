using FlexiGal
using LinearAlgebra
include("/home/hostos/Documents/FlexiGal/examples/Tetra_DC.jl");
model = create_model(xyz, conn);
dmax = 1.35;
ngpts = 4;
const Tc = 962.0; const vc = 0.001; const Tw = 303.0;
const kₛ = 231.0; const kₗ = 90.7; const cpₛ = 900.0; const cpₗ = 1300.0;
const ρₛ = 2700.0; const ρₗ = 2368.0; const Hf = 396400.0; const Tₗ = 930; const Tₛ = 919;

fₛ(T) = T <= Tₛ ? 1.0 : (T >= Tₗ ? 0.0 : 1.0 - (T - Tₛ) / (Tₗ - Tₛ))
gₛ(T) = fₛ(T)*ρₗ/(ρₗ + (1-fₛ(T))*(ρₛ-ρₗ))
function h(x)
    if x[3] < 0.13
        return 0.0                      # Adiabatic section
    elseif x[3] < 0.20 && x[3] >= 0.13
        return 1500.0                   # Mold contact
    elseif x[3] < 0.21 && x[3] >= 0.20
        return 150.0                    # Air gap
    elseif x[3] < 0.22 && x[3] >= 0.21
        return 150.0 + (x[3] - 0.21) * (19850.0 / 0.01) # Impinging jet: pendiente de 19,850 / 0.01 por cada unidad de z sobre 0.21
    elseif x[3] < 0.26 && x[3] >= 0.22
        return 20000.0 - (x[3] - 0.22) * (10000.0 / 0.04) # Boiling film: decreciente a razón de 10,000 / 0.04
    else
        return 10000.0                  # Stable falling film
    end
end
Bottom(x) = x[3] <= 0.0001;
Right(x) = x[1] >= 1.73/2 -1e-4
Front(x) = x[2] >= 0.66/2 -1e-4
set_boundary_tag!(model, Bottom; name="Bottom"); set_boundary_tag!(model, Right; name="Right");
set_boundary_tag!(model, Front; name="Front");
Ω = Triangulation(model, "Domain");
Γb = Triangulation(model, "Bottom"); Γr = Triangulation(model, "Right");
Γf = Triangulation(model, "Front");
dΩ = IntegrationSet(Ω, ngpts);
dΓr = IntegrationSet(Γr, ngpts-1); dΓf = IntegrationSet(Γf, ngpts-1);

Tspace = ApproxSpace(model, [Ω, Γb, Γr, Γf], Float64; dmax=dmax, method=:EFG, technique =:MK,
    Dirichlet_Boundaries=[Γb], Dirichlet_Values=[Tc],shape=:spherical, cell_based=true);

fspace = ApproxSpace(model, Ω, Float64; dmax=dmax, method=:EFG, technique =:MK,shape=:spherical, cell_based=true);

const v = VectorField(0.0, 0.0, vc);

@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρₛ * cpₛ * v ⋅ ∇(T)))dΩ + ∫(δT * (h * T))dΓr + ∫(δT * (h * T))dΓf
@WeakForm b(δT) = ∫(((δT * h) * Tw))dΓr + ∫(((δT * h) * Tw))dΓf
op = Linear_Problem(a, b, Tspace);
Th = Solve(op, dΩ);

function Deferred_Piccard(Th, op, fspace, Tspace; tol=1e-6, max_iter=60)
    _,_,Spaces=op
    Tspace_Built = Get_space_from_IntegrationSet(Spaces, dΩ)
    Measure = Get_Measures(Tspace_Built)
    Th_old = Th; Th_new = Th;
    gₛh = nothing;
    iter = 0; err = 1.0; erra = 1.0;
    k_nl(T) = gₛ(T) * kₛ + (1 - gₛ(T)) * kₗ
    cp_nl(T) = gₛ(T) * cpₛ + (1 - gₛ(T)) * cpₗ
    ρ_nl(T) = gₛ(T) * ρₛ + (1 - gₛ(T)) * ρₗ
    β = 1.0; ε = 1e-4;
    @WeakForm aₚ(δf, f) = ∫(δf * f + ε * ∇(δf) ⋅ (∇(f)))dΩ
    @WeakForm bₚ(δf) = ∫(δf * (gₛ ∘ Th))dΩ
    op_fs = Linear_Problem(aₚ, bₚ, fspace, op)
    while err > tol && iter < max_iter
        iter += 1
        gₛh = let Th = Th_old, fspace = fspace, op_fs=op_fs, dΩ=dΩ
        @WeakForm bₚ(δf) = ∫(δf * (gₛ ∘ Th))dΩ
        op_fs = Reassemble_Vector!(bₚ,fspace,op_fs)
        Solve(op_fs, dΩ)
        end
        Th_new = let Th_old = Th_old, gₛh = gₛh, Tspace = Tspace, op=op, dΩ=dΩ
           @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl ∘ Th_old * ∇(T)) + δT* (ρ_nl ∘ Th_old) * (cp_nl ∘ Th_old) * (v ⋅ ∇(T)))dΩ + 
           ∫(δT * (h * T))dΓr + ∫(δT * (h * T))dΓf
            @WeakForm b_picard(δT) =  ∫((δT * h) * Tw)dΓr + ∫((δT * h) * Tw)dΓf +  ∫(δT * (ρ_nl ∘ Th_old) * Hf * v ⋅ ∇(gₛh))dΩ
             op_nl = Linear_Problem(a_picard, b_picard, Tspace, op)
            @time Solve(op_nl, dΩ)
        end
        T_new = Get_Nodal_Values(Th_new)
        T_old = Get_Nodal_Values(Th_old)
        err = norm(T_new - T_old) / (norm(T_new))
        println("Picard Iteration $iter: Relative Error = $err")
        err > erra ? β /= 2 : erra = err
        @. T_old = T_old * (1 - β) + T_new * β
        β < 0.05 && (β = 1.0)
        Th_old = FlexiFunction(T_old, Tspace_Built, Measure[1])
    end
    return Th_new, gₛh
end

#open("resultados_picard.txt", "w") do io
 #   redirect_stdout(io) do
        #global 
        Th2, gₛh = Deferred_Piccard(Th, op, fspace, Tspace; max_iter=150, tol=1e-5)
 #   end
#end