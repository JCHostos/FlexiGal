using LinearAlgebra
struct EFGFunction
    fdom::Vector{Vector{Float64}}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Matrix{Float64}}
    tag::String
    Measure::Matrix{Float64}
end

function EFGFunction(field_nodal::Vector{Float64},
    Shape_Functions::EFGSpace,
    Measure::Tuple{String,Matrix{Float64}})
    tag, gs = Measure
    # Buscar funciones de forma para ese tag
    if haskey(Shape_Functions.domain, tag)
        PHI, DPHI, DOM = Shape_Functions.domain[tag]
    elseif haskey(Shape_Functions.boundary, tag)
        PHI, DPHI, DOM = Shape_Functions.boundary[tag]
    else
        error("No se encontraron funciones de forma para el tag '$tag'.")
    end
    # Construir Tdom en los puntos de Gauss
    ngauss = length(DOM)
    fdom = Vector{Vector{Float64}}(undef, ngauss)
    @inbounds for i in 1:ngauss
        fdom[i] = field_nodal[DOM[i]]
    end
    return EFGFunction(fdom, PHI, DPHI, tag, gs)
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
struct SingleEFGMeasure
    phi::Float64
    dphi::Vector{Float64}
end
@inline function SingleEFGMeasure(EFGm::EFGMeasure, ind::Int, a::Int)
    phi = EFGm.PHI[ind][a]
    dphi = EFGm.DPHI[ind][a, :]
    return SingleEFGMeasure(phi, dphi)
end
struct GradSingleEFGMeasure
    dphi::Vector{Float64}
end
@inline function ∇(SingleEFG::SingleEFGMeasure)
    return GradSingleEFGMeasure(SingleEFG.dphi)
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
    return dot(a.dphi,b.dphi)
end
struct Integrand
    object
end
const ∫ = Integrand
struct SingleDomainMeasure
    coordg::Vector{Float64}
    weight::Float64
    jacobian::Float64
end
# Domain Measure Operations
@inline function SingleDomainMeasure(Measure::EFGMeasure, ind::Int, dim::Int)
    gs = Measure.gs
    weight = gs[ind, end-1]
    jacobian = gs[ind, end]
    coordg = gs[ind, 1:dim]
    return SingleDomainMeasure(coordg,weight, jacobian)
end
# Integration Operations
function Integrate(a, b::EFGMeasure)
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a .* (jac .* weight)
end
function Integrate(a, b::SingleDomainMeasure)
    return a * (b.weight * b.jacobian)
end

import Base: *
(*)(a::Integrand, b::EFGMeasure) = Integrate(a.object, b)
(*)(a::Integrand, b::SingleDomainMeasure) = Integrate(a.object, b)
(*)(a::Float64, b::EFGFunction) = a*Get_Point_Values(b)
(*)(a::Float64, b::GradEFGFunction) = a*b.grads
(*)(a::Float64, b::SingleEFGMeasure) = a*b.phi
(*)(a::Float64, b::GradSingleEFGMeasure) = a*b.dphi
import Base: ⋅
(⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(⋅)(a::GradEFGFunction, b::GradEFGFunction) = Internal_Product(a, b)
(⋅)(a::SingleEFGMeasure, b::SingleEFGMeasure)= Internal_Product(a,b)
(⋅)(a::GradSingleEFGMeasure, b::GradSingleEFGMeasure)= Internal_Product(a,b)