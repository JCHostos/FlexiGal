using LinearAlgebra
struct EFGFunction
    fdom::Vector{Vector{Float64}}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Matrix{Float64}}
end

function EFGFunction(field_nodal::Vector{Float64},
    Space::EFGSpace,
    Measure::DomainMeasure)
    Shapes = EFG_Measure(Measure, Space)
    PHI, DPHI, DOM = Shapes.PHI, Shapes.DPHI, Shapes.DOM
    # Construir Tdom en los puntos de Gauss
    ngauss = length(DOM)
    fdom = Vector{Vector{Float64}}(undef, ngauss)
    @inbounds for i in 1:ngauss
        fdom[i] = field_nodal[DOM[i]]
    end
    return EFGFunction(fdom, PHI, DPHI)
end
# Operations over EFGFunctions
function Get_Point_Values(f::EFGFunction)
    ngauss = length(f.PHI)
    PHI = f.PHI
    fdom = f.fdom
    fs = Vector{Float64}(undef, ngauss)
    @inbounds for i in 1:ngauss
        fs[i] = dot(PHI[i], fdom[i])
    end
    return fs
end
struct GradEFGFunction
    grads::Vector{Vector{Float64}}
end
function ∇(f::EFGFunction)
    ngauss = length(f.DPHI)
    grads = Vector{Vector{Float64}}(undef, ngauss)
    @inbounds for i in 1:ngauss
        grads[i] = f.DPHI[i]' * f.fdom[i]
    end
    return GradEFGFunction(grads)
end
# Operations over EFGMeasures
struct GradEFGMeasure
    DPHI::Vector{Matrix{Float64}}
end
@inline function ∇(EFGm::EFGMeasure)
    return GradEFGMeasure(EFGm.DPHI)
end
@inline ∇(::Nothing) = nothing
struct SingleEFGMeasure
    phi::Float64
    dphi::Vector{Float64}
    ind::Int
    coord::Vector{Float64}
end
@inline function SingleEFGMeasure(EFGm::EFGMeasure, ind::Int, a::Int,coord::Vector{Float64})
    phi = EFGm.PHI[ind][a]
    dphi = EFGm.DPHI[ind][a, :]
    return SingleEFGMeasure(phi, dphi, ind, coord)
end
struct GradSingleEFGMeasure
    dphi::Vector{Float64}
    ind::Int
    coord::Vector{Float64}
end
@inline function ∇(SingleEFG::SingleEFGMeasure)
    return GradSingleEFGMeasure(SingleEFG.dphi,SingleEFG.ind,SingleEFG.coord)
end
function Internal_Product(a::EFGFunction, b::EFGFunction)
    as = Get_Point_Values(a)
    bs = Get_Point_Values(b)
    ns = length(as)
    product = Vector{Float64}(undef, ns)
    @inbounds for i in 1:ns
        product[i] = as[i] * bs[i]
    end
    return product
end
@inline function Get_Point_Value(a::EFGFunction, ind::Int)
    return dot(a.PHI[ind], a.fdom[ind])
end
@inline function Internal_Product(a::EFGFunction, b::SingleEFGMeasure)
    phi, dphi, ind, coord = b.phi, b.dphi, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return SingleEFGMeasure(as * phi, dphi, ind, coord)
end
@inline function Internal_Product(a::EFGFunction, b::GradSingleEFGMeasure)
    dphi, ind, coord = b.dphi, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return GradSingleEFGMeasure(as * dphi, ind, coord)
end
@inline function Internal_Product(a::Function, b::SingleEFGMeasure)
    phi, dphi, ind, coord = b.phi, b.dphi, b.ind, b.coord
    as = a(coord)
    return SingleEFGMeasure(as * phi, dphi, ind, coord)
end
@inline function Internal_Product(a::Function, b::GradSingleEFGMeasure)
    dphi, ind, coord = b.dphi, b.ind, b.coord
    as = a(coord)
    return GradSingleEFGMeasure(as * dphi, ind, coord)
end
function Internal_Product(a::GradEFGFunction, b::GradEFGFunction)
    as = a.grads
    bs = b.grads
    ns = length(as)
    product = Vector{Float64}(undef, ns)
    @inbounds for i in 1:ns
        product[i] = dot(as[i], bs[i])
    end
    return product
end
@inline function Internal_Product(a::SingleEFGMeasure, b::SingleEFGMeasure)
    return a.phi * b.phi
end
@inline function Internal_Product(a::GradSingleEFGMeasure, b::GradSingleEFGMeasure)
    return dot(a.dphi, b.dphi)
end
struct Integrand{T}
    object::T
end
const ∫ = Integrand
# Domain Measure Operations
# Integration Operations
# Caso 1: a.object es un vector numérico
@inline function Integrate(a::Integrand{<:AbstractVector}, b::DomainMeasure)
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a.object .* (jac .* weight)   # integración “vieja”
end

# Caso 2: a.object es cualquier otra cosa (función, simbólico, etc.)
@inline function Integrate(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}})
    measures = isa(b, DomainMeasure) ? [b] : b
    return (a.object, measures)
end
import Base: *
(*)(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}}) = Integrate(a, b)
(*)(a::Union{Float64,Int}, b::EFGFunction) = a * Get_Point_Values(b)
(*)(a::EFGFunction, b::Union{Float64,Int}) = Get_Point_Values(a) * b
(*)(a::Union{Float64,Int}, b::GradEFGFunction) = a * b.grads
(*)(a::GradEFGFunction, b::Union{Float64,Int}) = a.grads * b
(*)(a::Union{Float64,Int}, b::SingleEFGMeasure) = a * b.phi
(*)(a::SingleEFGMeasure, b::Union{Float64,Int}) = a.phi * b
(*)(a::Union{Float64,Int}, b::GradSingleEFGMeasure) = a * b.dphi
(*)(a::GradSingleEFGMeasure, b::Union{Float64,Int}) = a.dphi * b
(*)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(*)(a::SingleEFGMeasure, b::SingleEFGMeasure) = Internal_Product(a, b)
(*)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(*)(a::EFGFunction, b::GradSingleEFGMeasure) = Internal_Product(a, b)
(*)(a::Function, b::SingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Function, b::GradSingleEFGMeasure) = Internal_Product(a,b)
import Base: ⋅
(⋅)(a::Function, b::SingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::Function, b::GradSingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::EFGFunction, b::GradSingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(⋅)(a::GradEFGFunction, b::GradEFGFunction) = Internal_Product(a, b)
(⋅)(a::SingleEFGMeasure, b::SingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::GradSingleEFGMeasure, b::GradSingleEFGMeasure) = Internal_Product(a, b)
Internal_Product(a::Int, b::Int) = a * b
(⋅)(a::Int, b::Int) = Internal_Product(a, b)
Internal_Product(a::Int, b::Vector{Float64}) = a * b

# Nothing Operations
(*)(::EFGFunction,::Nothing)=nothing
(*)(::Function,::Nothing)=nothing
(*)(::Int,::Nothing)=nothing
(*)(::Float64,::Nothing)=nothing
(*)(::Nothing,::Nothing)=nothing
(⋅)(::Nothing,::Nothing)=nothing