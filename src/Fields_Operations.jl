using LinearAlgebra
function EFG_Field(field_nodal::Vector{Float64},
                   Shape_Functions::Dict,
                   Measure::Tuple{String,Matrix{Float64}})
    tag, gs = Measure

    # Buscar funciones de forma para ese tag
    if haskey(Shape_Functions[:domain], tag)
        PHI, DPHI, DOM = Shape_Functions[:domain][tag]
    elseif haskey(Shape_Functions[:boundary], tag)
        PHI, DPHI, DOM = Shape_Functions[:boundary][tag]
    else
        error("No se encontraron funciones de forma para el tag '$tag'.")
    end

    # Construir Tdom en los puntos de Gauss
    ngauss = length(DOM)
    Tdom = Vector{Vector{Float64}}(undef, ngauss)
    Ts = Vector{Float64}(undef, ngauss)
    @inbounds for i in 1:ngauss
        Tdom[i] = field_nodal[DOM[i]]
        Ts[i] = dot(PHI[i], Tdom[i])
    end
    return EFGFunction(Tdom, Ts, PHI, DPHI, tag, gs)
end
Get_Point_Values(f::EFGFunction) = f.Tgauss
function ∇(f::EFGFunction)
    ngauss = length(f.DPHI)
    grads  = Vector{Vector{Float64}}(undef, ngauss)
    @inbounds for i in 1:ngauss
        # gradiente = DPHI[i]' * Tdom[i]
        grads[i] = f.DPHI[i]' * f.Tdom[i]
    end
    return grads
end