using FlexiGal
using LinearAlgebra
using SparseArrays
using Arpack

# 1. Definición de Parámetros y Geometría
Lx = 1.0
Ly = 1.0
Domain = (Lx, Ly)
nd = 110
Divisions = (nd, nd)
dmax = 1.75
const Re = 500.0

# Funciones de clustering para concentrar nodos
function clustering(x)
    eps_val = 0.00001
    nx = 0.5 + 0.5 * tanh((2.0 * x[1] - 1.0) * eps_val) / tanh(eps_val)
    ny = 0.5 + 0.5 * tanh((2.0 * x[2] - 1.0) * eps_val) / tanh(eps_val)
    return [nx, ny]
end

function dm_clustered(x0)
    eps_val = 0.00001
    s1 = eps_val / (tanh(eps_val) * cosh((2.0 * x0[1] - 1.0) * eps_val)^2)
    s2 = eps_val / (tanh(eps_val) * cosh((2.0 * x0[2] - 1.0) * eps_val)^2)
    return [s1, s2]
end

# Creación del modelo y triangulaciones
model = create_model(Domain, Divisions; map=clustering)
ngpts = 3
ng2 = 1

Ω  = Triangulation(model, "Domain")
Γ1 = Triangulation(model, "Left")
Γ2 = Triangulation(model, "Bottom")
Γ3 = Triangulation(model, "Right")
Γ4 = Triangulation(model, "Top")

dΩ  = IntegrationSet(Ω, ngpts)
dΩₚ = IntegrationSet(Ω, ng2)

U₀(x)   = VectorField((4 * x[1] * (1 - x[1]))^(1/8), 0.0)
No_Slip = VectorField(0.0, 0.0)

# Espacios de Aproximación
Uspace = ApproxSpace(
    model, [Ω, Γ1, Γ2, Γ3, Γ4], VectorField{2, Float64};
    dmax, method=:EFG, technique=:MK, shape=:rectangular,
    Dirichlet_Boundaries=[Γ1, Γ2, Γ3, Γ4],
    Dirichlet_Values=[No_Slip, No_Slip, No_Slip, U₀],
    dm_map=dm_clustered
)

Pspace = ApproxSpace(model, [Ω], Float64; method=:FEM, order=0, constraint=:ZeroMean)
X = MultiApproxSpace(Uspace, Pspace)

# 2. Ensamblado del Problema Mixto Principal
@WeakForm a((δu, δp), (u, p)) = ∫(∇(δu) ⊙ (∇(u) + ∇(u)') * (1 / Re))dΩ - ∫((∇ ⋅ δu) * p - δp * (∇ ⋅ u))dΩₚ

println("Ensamblando problema lineal principal (A)...")
@time op = Linear_Problem(a, nothing, X)
A, F, _ = op

# 3. Ensamblado de la Matriz de Masa de Presión Exacta
@WeakForm mass_p(q, p) = ∫(q * p)dΩₚ

println("\nEnsamblando matriz de masa de presión (Mp)...")
op_Mp = Linear_Problem(mass_p, nothing, Pspace)
Mp = op_Mp[1]

# 4. Dimensiones y Separación de Submatrices
np = size(Mp, 1)          # Dimensión exacta de presión (nd * nd)
total_dofs = size(A, 1)   # GDL totales
nu = total_dofs - np      # GDL de velocidad

println("\n--- Dimensiones Verificadas ---")
println("GDL Totales:          $total_dofs")
println("GDL Velocidad (nu):  $nu")
println("GDL Presión (np):    $np")

# Extracción de subbloques Ku y B
Ku = A[1:nu, 1:nu]
B  = A[(nu+1):end, 1:nu]

# Filtrado de GDL de velocidad libres (eliminación de penalizaciones Dirichlet)
diag_Ku = diag(Ku)
max_diag = maximum(abs.(diag_Ku))
free_dofs = findall(x -> abs(x) < max_diag * 1e-3, diag_Ku)

if isempty(free_dofs)
    free_dofs = 1:nu
end

Ku_free = Ku[free_dofs, free_dofs]
B_free  = B[:, free_dofs]

# 5. Factorización LU y Construcción del Complemento de Schur S
println("\nFactorizando Ku_free con LU...")
@time F_Ku = lu(Ku_free)

println("Calculando S = B * Ku⁻¹ * Bᵀ...")
@time Ku_inv_Bt = F_Ku \ Matrix(B_free')
S = B_free * Ku_inv_Bt

# 6. Cálculo del Autovalor Inf-Sup con Arpack
nev_calc = 6
println("\nCalculando los $nev_calc menores autovalores usando Arpack (eigs)...")
@time vals, _ = eigs(S, Mp; nev=nev_calc, which=:SM)

# Ordenamos por magnitud
vals_pos = sort(abs.(real(vals)))

println("\n--- Muestra de Menores Autovalores ---")
display(vals_pos)

# Selección de λ_min (descartando el cero puramente numérico si hiciera falta)
λ_min = vals_pos[1] > 1e-6 ? vals_pos[1] : vals_pos[2]
γ_h = sqrt(λ_min)

println("\n------------------------------------------------")
println(" Constante inf-sup (γ_h) = ", γ_h)
println("------------------------------------------------")