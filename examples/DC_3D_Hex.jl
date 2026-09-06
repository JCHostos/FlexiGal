using FlexiGal
using LinearAlgebra
using DelimitedFiles
Lx, Ly, Lz = 1.73/2, 0.66/2, 2.5
Domain = (Lx,Ly,Lz); Divisions = (50, 28, 72)
conn=readdlm("/home/hostos/Documents/FlexiGal/examples/connDC.txt",Int);
xyz=readdlm("/home/hostos/Documents/FlexiGal/examples/xyzDC.txt",Float64);
model = create_model(xyz,conn);
model.domain = Domain; model.divisions = Divisions;
dmax = [1.5, 1.5, 1.05];
ngpts = 3;
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
        # Impinging jet: pendiente de 19,850 / 0.01 por cada unidad de z sobre 0.21
        return 150.0 + (x[3] - 0.21) * (19850.0 / 0.01)
    elseif x[3] < 0.26 && x[3] >= 0.22
        # Boiling film: decreciente a razón de 10,000 / 0.04
        return 20000.0 - (x[3] - 0.22) * (10000.0 / 0.04)
    else
        return 10000.0                  # Stable falling film
    end
end

z_int = 0.75
MK(x)  = x[3] <= z_int
FEM(x) = !MK(x)
Bottom(x) = x[3] <= 0.0001;
Right(x) = x[1] >= 1.73/2 -1e-4
Front(x) = x[2] >= 0.66/2 -1e-4
set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
set_boundary_tag!(model, Bottom; name="Bottom"); set_boundary_tag!(model, Right; name="Right");
set_boundary_tag!(model, Front; name="Front");
Ω_fem = Triangulation(model, "FEM"); Ω_mk = Triangulation(model, "MK");
Γb = Triangulation(model, "Bottom"); Γr = Triangulation(model, "Right");
Γf = Triangulation(model, "Front");
dΩ_fem = IntegrationSet(Ω_fem, ngpts-1); dΩ_mk = IntegrationSet(Ω_mk, ngpts);
dΓr = IntegrationSet(Γr, ngpts); dΓf = IntegrationSet(Γf, ngpts);

Tspace = ApproxSpace(model, [Ω_fem, Ω_mk, Γb, Γr, Γf], Float64; dmax, method=[:FEM,:EFG,:FEM,:FEM,:FEM], technique =:MK,
    Dirichlet_Boundaries=[Γb], Dirichlet_Values=[Tc],shape=:rectangular);

fspace = ApproxSpace(model, [Ω_fem, Ω_mk], Float64; dmax, method=[:FEM,:EFG], technique =:MK,shape=:rectangular);

const v = VectorField(0.0, 0.0, vc);

@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρₛ * cpₛ * v ⋅ ∇(T)))dΩ_fem + 
∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρₛ * cpₛ * v ⋅ ∇(T)))dΩ_mk + ∫(δT * (h * T))dΓr + ∫(δT * (h * T))dΓf
@WeakForm b(δT) = ∫(((δT * h) * Tw))dΓr + ∫(((δT * h) * Tw))dΓf
op = Linear_Problem(a, b, Tspace);
GC.gc()
Th = Solve(op);

function Deferred_Piccard(Th, op, fspace, Tspace; tol=1e-6, max_iter=60)
    _,_,Spaces=op
    Tspace_Built_fem = Get_space_from_IntegrationSet(Spaces, dΩ_fem)
    Tspace_Built_mk = Get_space_from_IntegrationSet(Spaces, dΩ_mk)
    Measure_fem = Get_Measures(Tspace_Built_fem)
    Measure_mk = Get_Measures(Tspace_Built_mk)
    Th_old = Th; Th_new = Th;
    gₛh = nothing;
    iter = 0; err = 1.0; erra = 1.0;
    k_nl(T) = gₛ(T) * kₛ + (1 - gₛ(T)) * kₗ
    cp_nl(T) = gₛ(T) * cpₛ + (1 - gₛ(T)) * cpₗ
    ρ_nl(T) = gₛ(T) * ρₛ + (1 - gₛ(T)) * ρₗ
    β = 1.0; ε = 1e-4;
    Th_fem=Th[dΩ_fem]; Th_mk=Th[dΩ_mk]
    @WeakForm aₚ(δf, f) = ∫(δf * f + ε * ∇(δf) ⋅ (∇(f)))dΩ_fem + ∫(δf * f + ε * ∇(δf) ⋅ (∇(f)))dΩ_mk
    @WeakForm bₚ(δf) = ∫(δf * (gₛ ∘ Th_fem))dΩ_fem + ∫(δf * (gₛ ∘ Th_mk))dΩ_mk
    op_fs = Linear_Problem(aₚ, bₚ, fspace, op)
    GC.gc()
    while err > tol && iter < max_iter
        iter += 1
        gₛh = let Th = Th_old, fspace = fspace, op_fs=op_fs, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
        Th_mk = Th[dΩ_mk]; Th_fem = Th[dΩ_fem]
        @WeakForm bₚ(δf) = ∫(δf * (gₛ ∘ Th_fem))dΩ_fem + ∫(δf * (gₛ ∘ Th_mk))dΩ_mk
        op_fs = Reassemble_Vector!(bₚ,fspace,op_fs)
        Solve(op_fs)
        end
        Th_new = let Th_old = Th_old, gₛh = gₛh, Tspace = Tspace, op=op, dΩ_fem=dΩ_fem, dΩ_mk=dΩ_mk
            gₛh_fem=gₛh[dΩ_fem]; gₛh_mk=gₛh[dΩ_mk]
            Th_old_fem=Th_old[dΩ_fem]; Th_old_mk=Th_old[dΩ_mk]
           @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl ∘ Th_old_fem * ∇(T)) + δT* (ρ_nl ∘ Th_old_fem) * (cp_nl ∘ Th_old_fem) * (v ⋅ ∇(T)))dΩ_fem + 
            ∫(∇(δT) ⋅ (k_nl ∘ Th_old_mk * ∇(T)) + δT * (ρ_nl ∘ Th_old_mk) * (cp_nl ∘ Th_old_mk) * (v ⋅ ∇(T)))dΩ_mk  + ∫(δT * (h * T))dΓr + ∫(δT * (h * T))dΓf
            @WeakForm b_picard(δT) =  ∫((δT * h) * Tw)dΓr + ∫((δT * h) * Tw)dΓf +  ∫(δT * (ρ_nl ∘ Th_old_fem) * Hf * v ⋅ ∇(gₛh_fem))dΩ_fem + ∫(δT * (ρ_nl ∘ Th_old_mk)* Hf * v ⋅ ∇(gₛh_mk))dΩ_mk
            @time op_nl = Linear_Problem(a_picard, b_picard, Tspace, op)
            Solve(op_nl)
        end
        T_new = Get_Nodal_Values(Th_new[dΩ_fem])
        T_old = Get_Nodal_Values(Th_old[dΩ_fem])
        err = norm(T_new - T_old) / (norm(T_new))
        println("Picard Iteration $iter: Relative Error = $err")
        err > erra ? β /= 2 : erra = err
        @. T_old = T_old * (1 - β) + T_new * β
        β < 0.05 && (β = 1.0)
        Th_old = Dict(dΩ_fem => FlexiFunction(T_old, Tspace_Built_fem, Measure_fem[1]),
        dΩ_mk  => FlexiFunction(T_old, Tspace_Built_mk, Measure_mk[1]))
    end
    return Th_new, gₛh
end
Th2, gₛh = Deferred_Piccard(Th, op, fspace, Tspace; max_iter=150, tol=1e-5);

Tvals=Get_Nodal_Values(Th2[dΩ_fem])
Ω = Triangulation(model, "Domain");
dΩ = IntegrationSet(Ω, ngpts);
Tspaceᵢ = ApproxSpace(model, [Ω], Float64; dmax, method=:FEM, technique =:MK,shape=:spherical, cell_based=true);
Thᵢ=FlexiFunction(Tvals,Tspaceᵢ,dΩ);
@WeakForm aᵣ(δT, T) = ∫(δT * T)dΩ
@WeakForm bᵣ(δT) = ∫(δT * Thᵢ)dΩ
Tspaceᵣ = ApproxSpace(model, [Ω], Float64; dmax=1.35, method=:EFG, technique =:IMLS,shape=:rectangular, cell_based=true);
opᵣ=Linear_Problem(aᵣ,bᵣ,Tspaceᵣ)
Thᵣ = Solve(opᵣ,dΩ)
Tvals2=Get_Nodal_Values(Thᵣ)
Tspaceᵣ = ApproxSpace(model, [Ω], Float64; dmax=1.7*dmax, technique =:IMLS,shape=:spherical, cell_based=true);
Thn=EFGInterpolatedField(Tvals2,Tspaceᵣ,model.x)
Tf=Get_Interpolated_Values(Thn)