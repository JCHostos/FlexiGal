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
function ∇(EFGm::EFGMeasure)
    return GradEFGMeasure(EFGm.DPHI)
end
function Point_Node_EFGMeasure(EFGm::EFGMeasure, ind::Int, a::Int)
    return EFGm.PHI[ind][a]
end
function Point_Node_GradEFGMeasure(GradEFGm::GradEFGMeasure, ind::Int, a::Int)
    return GradEFGm.DPHI[ind][a, :]
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
struct Integrand
    object
end
const ∫ = Integrand
struct SingleDomainMeasure
weight::Float64
jacobian::Float64
coordg::Vector{Float64}
end
# Domain Measure Operations
function SingleDomainMeasure(Measure::DomainMeasure, ind::Int,dim::Int)
    gs = Measure.gs
    weight = gs[ind, end-1]
    jacobian = gs[ind, end]
    coordg = gs[ind,1:dim]
    return SingleDomainMeasure(coordg,weight, jacobian)
end
# Integration Operations
function Integrate(a, b::DomainMeasure)
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a .* (jac .* weight)
end
function Integrate(a,b::SingleDomainMeasure)
    return a * (b.weight * b.jacobian)
end
import Base: *
(*)(a::Integrand, b::DomainMeasure) = Integrate(a.object, b)
(*)(a::Integrand, b::SingleDomainMeasure) = Integrate(a.object, b)
import Base: ⋅
(⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
(⋅)(a::GradEFGFunction, b::GradEFGFunction) = Internal_Product(a, b)
