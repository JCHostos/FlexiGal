using LinearAlgebra
using SparseArrays
using Arpack

@WeakForm mass_p(q, p) = ∫(q * p)dΩ
println("Ensamblando matriz de masa de presión Mp...")
op_Mp = Linear_Problem(mass_p, nothing, Pspace)
Mp = op_Mp[1]

# 4. Dimensiones e Identificación de Submatrices
np = size(Mp, 1)          # Dimensión exacta de presión (nd * nd = 4900)
total_dofs = size(A, 1)   # GDL totales
nu = total_dofs - np      # GDL de velocidad

println("\n--- Dimensiones Verificadas ---")
println("GDL Totales:          $total_dofs")
println("GDL Velocidad (nu):  $nu")
println("GDL Presión (np):    $np")

# Submatrices de la forma débil
Ku = A[1:nu, 1:nu]
B  = A[(nu+1):end, 1:nu]

# Filtrado de GDL de velocidad libres (penalización)
diag_Ku = diag(Ku)
max_diag = maximum(abs.(diag_Ku))
free_dofs = findall(x -> abs(x) < max_diag * 1e-3, diag_Ku)

if isempty(free_dofs)
    free_dofs = 1:nu
end

Ku_free = Ku[free_dofs, free_dofs]
B_free  = B[:, free_dofs]

# 5. Factorización y Complemento de Schur S
println("\nFactorizando Ku_free con LU...")
@time F_Ku = lu(Ku_free)

println("Calculando S = B * Ku⁻¹ * Bᵀ ...")
@time Ku_inv_Bt = F_Ku \ Matrix(B_free')
S = B_free * Ku_inv_Bt

# 6. Cálculo Eficiente de Autovalores Parciales con Arpack
nev_calc = 6
println("Calculando los $nev_calc menores autovalores usando Arpack (eigs)...")
@time vals, _ = eigs(S, Mp; nev=nev_calc, which=:SM)

# Ordenar autovalores
vals_pos = sort(abs.(real(vals)))

println("\n--- Muestra de Menores Autovalores ---")
display(vals_pos)

# Selección de autovalor inf-sup (λ_min)
# Si :ZeroMean eliminó el modo constante, el primer valor (vals_pos[1]) es directo el LBB
λ_min = vals_pos[1] > 1e-6 ? vals_pos[1] : vals_pos[2]
γ_h = sqrt(λ_min)

println("\n------------------------------------------------")
println(" Constante inf-sup (γ_h) = ", γ_h)
println("------------------------------------------------")