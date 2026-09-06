using FlexiGal
using LinearAlgebra
include("/home/hostos/Documents/FlexiGal/examples/Triangles_SquareCav_CircObs_Pe20.jl");
dmax = 2.0;
ngpts = 3;
model = create_model(xy, conn);
const Pe=20.0;
const TL = 899.0; const TR = 950.0;
const kₛ = 231.0; const kₗ = 90.7; const cpₛ = 900.0; const cpₗ = 1300.0;
const ρₛ = 2700.0; const ρₗ = 2368 ; const Hf = 396400.0; const Tₗ = 930; const Tₛ = 919;
const u0=Pe*kₗ/(ρₗ*cpₗ)
fₛ(T) = T <= Tₛ ? 1.0 : (T >= Tₗ ? 0.0 : 1.0 - (T - Tₛ) / (Tₗ - Tₛ))
gₛ(T) = fₛ(T)*ρₗ/(ρₗ + (1-fₛ(T))*(ρₛ-ρₗ))
MK(x) = x[1] >= -0.35 && x[1]<0.45; 
FEM(x)  = !MK(x);
set_tag!(model, MK; name="MK"); set_tag!(model, FEM; name="FEM");
Right(x) = (x[1] > 0.5-1e-4)
Left(x) = x[1] < -0.5+1e-4
Top(x) = (x[2] > 0.5-1e-4)
Bottom(x) = x[2] < -0.5+1e-4
Ω_fem = Triangulation(model, "FEM"); Ω_mk = Triangulation(model, "MK");
set_boundary_tag!(model, Left; name="Left"); set_boundary_tag!(model, Right; name="Right");
set_boundary_tag!(model, Top; name="Top"); set_boundary_tag!(model, Bottom; name="Bottom");
ΓL = Triangulation(model, "Left"); ΓR = Triangulation(model, "Right");
ΓT = Triangulation(model, "Top"); ΓB = Triangulation(model, "Bottom");
dΩ_fem = IntegrationSet(Ω_fem, ngpts); dΩ_mk = IntegrationSet(Ω_mk, ngpts);

Tspace = ApproxSpace(model, [Ω_fem, Ω_mk, ΓL, ΓR, ΓT, ΓB], Float64; dmax, method=[:FEM,:EFG,:FEM,:FEM,:FEM,:FEM], technique =:LME,shape=:circular, 
cell_based=true, Dirichlet_Boundaries=[ΓL,ΓR,ΓT,ΓB], Dirichlet_Values=[TL,TR,x->(TL+TR)/2+(TR-TL)*x[1],x->(TL+TR)/2+(TR-TL)*x[1]]);

fspace = ApproxSpace(model, [Ω_fem, Ω_mk, ΓL, ΓR, ΓT, ΓB], Float64; dmax, method=[:FEM,:EFG,:FEM,:FEM,:FEM,:FEM], technique =:LME, shape=:circular, 
cell_based=true);
const v(x) = VectorField(-u0 * x[2], u0 * x[1])
@WeakForm a(δT, T) = ∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρₛ * cpₛ * v ⋅ ∇(T)))dΩ_fem + 
∫(∇(δT) ⋅ (kₛ * ∇(T)) + δT * (ρₛ * cpₛ * v ⋅ ∇(T)))dΩ_mk
op = Linear_Problem(a,Tspace);
Th=Solve(op)
function Deferred_Piccard(Th, op, fspace, Tspace; tol=1e-6, max_iter=60)
    _,_,Spaces=op
    Tspace_Built_fem = Get_space_from_IntegrationSet(Spaces, dΩ_fem)
    Tspace_Built_mk = Get_space_from_IntegrationSet(Spaces, dΩ_mk)
    Measure_fem = Get_Measures(Tspace_Built_fem)
    Measure_mk = Get_Measures(Tspace_Built_mk)
    Th_old = Th; Th_new = Th;
    gₛh = nothing
    iter = 0; err = 1.0; erra = 1.0;
    k_nl(T) = gₛ(T) * kₛ + (1 - gₛ(T)) * kₗ
    cp_nl(T) = gₛ(T) * cpₛ + (1 - gₛ(T)) * cpₗ
    ρ_nl(T) = gₛ(T) * ρₛ + (1 - gₛ(T)) * ρₗ
    β = 1.0; ε = 1e-9;
    Th_fem=Th[dΩ_fem]; Th_mk=Th[dΩ_mk]
    @WeakForm aₚ(δf, f) = ∫(δf * f + ε * ∇(δf) ⋅ (∇(f)))dΩ_fem + ∫(δf * f + ε * ∇(δf) ⋅ ∇(f))dΩ_mk
    @WeakForm bₚ(δf) = ∫(δf * (gₛ ∘ Th_fem))dΩ_fem + ∫(δf * (gₛ ∘ Th_mk))dΩ_mk
    op_fs = Linear_Problem(aₚ, bₚ, fspace, op)
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
            @WeakForm a_picard(δT, T) = ∫(∇(δT) ⋅ (k_nl ∘ Th_old_fem * ∇(T)) + (δT *(1-gₛh_fem))* (ρ_nl ∘ Th_old_fem) * (cp_nl ∘ Th_old_fem) * (v ⋅ ∇(T)))dΩ_fem + 
            ∫(∇(δT) ⋅ (k_nl ∘ Th_old_mk * ∇(T)) + (δT *(1-gₛh_mk)) * (ρ_nl ∘ Th_old_mk) * (cp_nl ∘ Th_old_mk) * (v ⋅ ∇(T)))dΩ_mk
            @WeakForm b_picard(δT) = ∫((δT *(1-gₛh_fem)) * (ρ_nl ∘ Th_old_fem) * Hf * v ⋅ ∇(gₛh_fem))dΩ_fem + ∫((δT *(1-gₛh_mk)) * (ρ_nl ∘ Th_old_mk)* Hf * v ⋅ ∇(gₛh_mk))dΩ_mk
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
Th2,gₛh = Deferred_Piccard(Th, op, fspace, Tspace; max_iter=150, tol=1e-5);
@WeakForm a2(δT, T) = ∫(δT*T)dΩ_fem + ∫(δT*T)dΩ_mk
@WeakForm b2(δT) = ∫(δT*Th2[dΩ_fem])dΩ_fem + ∫(δT*Th2[dΩ_mk])dΩ_mk
Ω = Triangulation(model, "Domain");
dΩ = IntegrationSet(Ω, ngpts);
Tspaceᵢ = ApproxSpace(model, [Ω], Float64; dmax=dmax*1.5, method=:EFG, technique =:LME,shape=:circular, cell_based=true);
op2 = Linear_Problem(a2, b2, Tspaceᵢ);
Th3 =Solve(op2);
Tvals = Get_Nodal_Values(Th3[dΩ_fem]);
include("/home/hostos/Documents/FlexiGal/examples/Triang_Cav_CirObs.jl"); # de aquí salen los últimos xy y conn para postprocesamiento
model1 = create_model(xy, conn);
Thn=EFGInterpolatedField(Tvals,Tspaceᵢ,model1.x)