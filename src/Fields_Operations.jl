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

struct VecEFGFunction
    fdom::Vector{Vector{Float64}}
    VEC::Vector{Matrix{Float64}}
end
∇(f::EFGFunction) = VecEFGFunction(f.fdom, f.DPHI)
function Get_Point_Values(f::VecEFGFunction)
    ngauss = length(f.VEC)
    VEC = f.VEC
    fdom = f.fdom
    gradfs = Vector{Vector{Float64}}(undef, ngauss)
    @inbounds for i in 1:ngauss
        gradfs[i] = VEC[i]'*fdom[i]
    end
    return gradfs
end

# Operations over EFGMeasures
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
struct VecSingleEFGMeasure
    vec::Vector{Float64}
    ind::Int
    coord::Vector{Float64}
end
@inline function ∇(SingleEFG::SingleEFGMeasure)
    return VecSingleEFGMeasure(SingleEFG.dphi,SingleEFG.ind,SingleEFG.coord)
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
@inline function Get_Point_Value(a::VecEFGFunction, ind::Int)
    return a.VEC[ind]'*a.fdom[ind]
end
@inline function Internal_Product(a::EFGFunction, b::SingleEFGMeasure)
    phi, dphi, ind, coord = b.phi, b.dphi, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return SingleEFGMeasure(as * phi, dphi, ind, coord)
end
@inline function Internal_Product(a::VecEFGFunction, b::SingleEFGMeasure)
    phi, ind, coord = b.phi, b.ind, b.coord
    as = Get_Point_Value(a,ind)
    return VecSingleEFGMeasure(as*phi, ind, coord)
end
@inline function Internal_Product(a::EFGFunction, b::VecSingleEFGMeasure)
    vec, ind, coord = b.vec, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return VecSingleEFGMeasure(as*vec, ind, coord)
end
@inline function Internal_Product(a::VecEFGFunction, b::VecSingleEFGMeasure)
    vec, ind, coord = b.vec, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return SingleEFGMeasure(dot(as,b.vec),vec, ind, coord)
end
@inline function Internal_Product(a::Function, b::SingleEFGMeasure)
    phi, dphi, ind, coord = b.phi, b.dphi, b.ind, b.coord
    as = a(coord)
    return SingleEFGMeasure(as * phi, dphi, ind, coord)
end
@inline function Internal_Product(a::Function, b::VecSingleEFGMeasure)
    as = a(b.coord)  # puede ser escalar o VectorField
    if isa(as, Number)
        # a(coord) devuelve escalar
        return VecSingleEFGMeasure(as * b.vec, b.ind, b.coord)
    elseif isa(as, VectorField)
        # a(coord) devuelve un vector: producto escalar
        value = sum(as[i] * b.vec[i] for i in 1:length(b.vec))
        return SingleEFGMeasure(value, b.vec, b.ind, b.coord)
    else
        error("Internal_Product: valor de retorno inesperado")
    end
end
function Internal_Product(a::VecEFGFunction, b::VecEFGFunction)
    as = Get_Point_Values(a)
    bs = Get_Point_Values(b)
    ns = length(as)
    product = Vector{Float64}(undef, ns)
    @inbounds for i in 1:ns
        product[i] = dot(as[i], bs[i])
    end
    return product
end

struct Composition{T<:Union{EFGFunction, VecEFGFunction}}
    f::Function
    Th::T
end

# --------------------------------------------------
# Constructores con sobrecarga de ∘ (\circ)
# --------------------------------------------------
import Base: ∘

# Function ∘ EFGFunction → Composition
(∘)(a::Function, b::EFGFunction) = Composition{EFGFunction}(a, b)
(∘)(a::EFGFunction, b::Function) = Composition{EFGFunction}(b, a)

# Function ∘ VecEFGFunction → Composition
(∘)(a::Function, b::VecEFGFunction) = Composition{VecEFGFunction}(a, b)
(∘)(a::VecEFGFunction, b::Function) = Composition{VecEFGFunction}(b, a)

# --------------------------------------------------
# Evaluación en todos los puntos
# --------------------------------------------------
function Get_Point_Values(c::Composition{EFGFunction})
    vals = Get_Point_Values(c.Th)
    return c.f.(vals)  # aplica f escalar a cada valor
end

function Get_Point_Values(c::Composition{VecEFGFunction})
    vals = Get_Point_Values(c.Th)
    return [c.f(v) for v in vals]  # aplica f a cada vector
end

# --------------------------------------------------
# Evaluación en un punto específico
# --------------------------------------------------
function Get_Point_Value(c::Composition{EFGFunction}, ind::Int)
    val = Get_Point_Value(c.Th, ind)
    return c.f(val)
end

function Get_Point_Value(c::Composition{VecEFGFunction}, ind::Int)
    val = Get_Point_Value(c.Th, ind)
    return c.f(val)
end

@inline function Internal_Product(k::Composition{EFGFunction}, b::SingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)  # obtiene T en ese punto de Gauss
    return SingleEFGMeasure(val*b.phi, b.dphi, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{EFGFunction}, b::VecSingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)
    return VecSingleEFGMeasure(val*b.vec, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{VecEFGFunction}, b::SingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)  # obtiene T en ese punto de Gauss
    return SingleEFGMeasure(val*b.phi, b.dphi, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{VecEFGFunction}, b::VecSingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)
    return VecSingleEFGMeasure(val*b.vec, b.ind, b.coord)
end

struct VectorField{D,T}
    data::NTuple{D,T}
end

# Constructor rápido
VectorField(x::Vararg{T,D}) where {T,D} = VectorField{D,T}(x)

# Acceso tipo array
Base.getindex(v::VectorField{D,T}, i::Int) where {D,T} = v.data[i]
Base.length(::VectorField{D,T}) where {D,T} = D

struct Integrand{T}
    object::T
end
const ∫ = Integrand
# Domain Measure Operations
# Integration Operations
# Caso 1: a.object es un vector numérico
function Integrate(a::Integrand{<:AbstractVector}, b::DomainMeasure)
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a.object .* (jac .* weight)   # integración “vieja”
end

@inline function Integrate(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}})
    measures = isa(b, DomainMeasure) ? [b] : b
    return (a.object, measures)
end

import Base: *
()(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}}) = Integrate(a, b)
(*)(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}}) = Integrate(a, b)
(*)(a::Union{Float64,Int}, b::EFGFunction) = a * Get_Point_Values(b)
(*)(a::EFGFunction, b::Union{Float64,Int}) = Get_Point_Values(a) * b
(*)(a::Union{Float64,Int}, b::SingleEFGMeasure) =  SingleEFGMeasure(a*b.phi,b.dphi,b.ind,b.coord)
(*)(a::SingleEFGMeasure, b::Union{Float64,Int}) = SingleEFGMeasure(b*a.phi,a.dphi,a.ind,a.coord)
(*)(a::Union{Float64,Int}, b::VecSingleEFGMeasure) = VecSingleEFGMeasure(a * b.vec,b.ind,b.coord) 
(*)(a::VecSingleEFGMeasure, b::Union{Float64,Int}) = VecSingleEFGMeasure(b * a.vec,a.ind,a.coord)
(*)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(*)(a::SingleEFGMeasure, b::SingleEFGMeasure) = a.phi*b.phi
(*)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(*)(a::EFGFunction, b::VecSingleEFGMeasure) = Internal_Product(a, b)
(*)(a::VecEFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(*)(a::Function, b::SingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Function, b::VecSingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Composition{EFGFunction}, b::SingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Composition{EFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Composition{VecEFGFunction}, b::SingleEFGMeasure) = Internal_Product(a,b)
(*)(a::Composition{VecEFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a,b)
import Base: ⋅
(⋅)(a::Function, b::SingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::Function, b::VecSingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::EFGFunction, b::VecSingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::VecEFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
(⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(⋅)(a::VecEFGFunction, b::VecEFGFunction) = Internal_Product(a, b)
(⋅)(a::SingleEFGMeasure, b::SingleEFGMeasure) = a.phi*b.phi
(⋅)(a::VecSingleEFGMeasure, b::VecSingleEFGMeasure) = dot(a.vec, b.vec)
(⋅)(a::Composition{EFGFunction}, b::SingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::Composition{EFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::Composition{VecEFGFunction}, b::SingleEFGMeasure) = Internal_Product(a,b)
(⋅)(a::Composition{VecEFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a,b)
(⋅)(v::VectorField{D,T}, g::VecSingleEFGMeasure) where {D,T} =
SingleEFGMeasure(sum(v[i] * g.vec[i] for i in 1:D), g.vec, g.ind, g.coord)
(⋅)(a::Int, b::Int) = a*b
(⋅)(a::Int, b::Vector{Float64}) = a * b
(⋅)(a::Vector{Float64},b::Int) = a * b
# Nothing Operations
(*)(::Nothing,::Nothing)=nothing
(⋅)(::Nothing,::Nothing)=nothing
(*)(::Any,::Nothing) = nothing
(⋅)(::Any,::Nothing) = nothing

import Base: +
(+)(::Nothing,::Nothing) = nothing
import Base: -
(-)(a::Nothing,b::Nothing) = nothing