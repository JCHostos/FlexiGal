using LinearAlgebra
struct EFGFunction
    fdom::Union{Vector{Vector{Float64}},Vector{Matrix{Float64}}}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Matrix{Float64}}
    D::Int
    Measure::DomainMeasure
end

@inline function EFGFunction(field_nodal::Vector{Float64},
    Space::EFGSpace,
    Measure::DomainMeasure)

    Shapes = EFG_Measure(Measure, Space)
    PHI, DPHI, DOM = Shapes.PHI, Shapes.DPHI, Shapes.DOM

    D = field_dim(Space.Field_Type)
    ngauss = length(DOM)

    if D == 1
        # --- Scalar Case: fdom is Vector{Vector{Float64}} ---
        fdom_scalar = Vector{Vector{Float64}}(undef, ngauss)
        @inbounds for i in 1:ngauss
            fdom_scalar[i] = field_nodal[DOM[i]]
        end
        return EFGFunction(fdom_scalar, PHI, DPHI, D, Measure)
    else
        # --- Vector Case: fdom is Vector{Matrix{Float64}} ---
        fdom_vectorial = Vector{Matrix{Float64}}(undef, ngauss)
        @inbounds for i in 1:ngauss
            dom = DOM[i]
            nlocal = length(dom)
            local_mat = Matrix{Float64}(undef, nlocal, D)
            for a in 1:nlocal
                global_node = dom[a]
                for d in 1:D
                    local_mat[a, d] = field_nodal[(global_node-1)*D+d]
                end
            end
            fdom_vectorial[i] = local_mat
        end
        return EFGFunction(fdom_vectorial, PHI, DPHI, D, Measure)
    end
end

# Operations over EFGFunctions
@inline function Get_Point_Values(f::EFGFunction)
    ngauss = length(f.PHI)

    if f.D == 1
        # Scalar: Returns Vector{Float64}
        fs = Vector{Float64}(undef, ngauss)
        @inbounds for i in 1:ngauss
            fs[i] = dot(f.PHI[i], f.fdom[i])
        end
        return fs
    else
        # Vector: Returns Vector{Vector{Float64}}
        vs = Vector{Vector{Float64}}(undef, ngauss)
        @inbounds for i in 1:ngauss
            vs[i] = [dot(f.PHI[i], f.fdom[i][:, d]) for d in 1:f.D]
        end
        return vs
    end
end

struct VecEFGFunction
    fdom::Union{Vector{Vector{Float64}},Vector{Matrix{Float64}}}
    VEC::Vector{Matrix{Float64}}
    D::Int
    Measure::DomainMeasure
end

struct Nabla end
const ∇ = Nabla()

@inline function (∇::Nabla)(f::EFGFunction)
    return VecEFGFunction(f.fdom, f.DPHI, f.D, f.Measure)
end
@inline function Get_Point_Values(f::VecEFGFunction)
    ngauss = length(f.VEC)
    if f.D == 1
        gradfs = Vector{Vector{Float64}}(undef, ngauss)
        @inbounds for i in 1:ngauss
            gradfs[i] = f.VEC[i]' * f.fdom[i]
        end
        return gradfs
    else
        gradfs_vec = Vector{Matrix{Float64}}(undef, ngauss)
        @inbounds for i in 1:ngauss
            gradfs_vec[i] = f.fdom[i]' * f.VEC[i]
        end
        return gradfs_vec
    end
end
struct TransposedVecEFGFunction
    origin::VecEFGFunction
end

struct VolumetricEFGFunction
    origin::EFGFunction
end

struct CompositeEFGFunction
    parts::Vector{Any}
    coeffs::Vector{Float64}
end

import Base: adjoint

@inline adjoint(f::VecEFGFunction) = TransposedVecEFGFunction(f)

@inline Get_Point_Values(f::TransposedVecEFGFunction) = [Matrix(m') for m in Get_Point_Values(f.origin)]

@inline function Get_Point_Values(f::VolumetricEFGFunction)
    H_vals = Get_Point_Values(∇(f.origin))
    D = f.origin.D
    # Matrix{Float64}(I, D, D) es más seguro y explícito
    Id = Matrix{Float64}(I, D, D) 
    return [tr(H) * Id for H in H_vals]
end

@inline function Get_Point_Values(f::CompositeEFGFunction)
    vals_part1 = Get_Point_Values(f.parts[1])
    res = f.coeffs[1] .* vals_part1
    for i in 2:length(f.parts)
        res .+= f.coeffs[i] .* Get_Point_Values(f.parts[i])
    end
    return res
end

@inline function Get_Point_Value(a::EFGFunction, ind::Int)
    return dot(a.PHI[ind], a.fdom[ind])
end
@inline function Get_Point_Value(a::VecEFGFunction, ind::Int)
    if a.D == 1
        return a.VEC[ind]' * a.fdom[ind]
    else
        return a.fdom[ind]' * a.VEC[ind]
    end
end

@inline (∇::Nabla)(::Nothing) = nothing

# Operations over EFGMeasures
struct SingleEFGMeasure
    phi::Float64
    dphi::Vector{Float64}
    ind::Int
    coord::Vector{Float64}
end
@inline function SingleEFGMeasure(EFGm::EFGMeasure, ind::Int, a::Int, coord::Vector{Float64})
    phi = EFGm.PHI[ind][a]
    dphi = EFGm.DPHI[ind][a, :]
    return SingleEFGMeasure(phi, dphi, ind, coord)
end
struct VecSingleEFGMeasure
    vec::Vector{Float64}
    ind::Int
    coord::Vector{Float64}
end
@inline function (∇::Nabla)(SingleEFG::SingleEFGMeasure)
    return VecSingleEFGMeasure(SingleEFG.dphi, SingleEFG.ind, SingleEFG.coord)
end

@inline function Internal_Product(a::EFGFunction, b::EFGFunction)
    as = Get_Point_Values(a)
    bs = Get_Point_Values(b)
    ns = length(as)
    product = Vector{Float64}(undef, ns)
    @inbounds for i in 1:ns
        product[i] = as[i] * bs[i]
    end
    return product
end

@inline function Internal_Product(a::VecEFGFunction, b::VecEFGFunction)
    as = Get_Point_Values(a)
    bs = Get_Point_Values(b)
    ns = length(as)
    product = Vector{Float64}(undef, ns)
    @inbounds for i in 1:ns
        product[i] = dot(as[i], bs[i])
    end
    return product
end

# EFGFunctions-SingleMeasures
@inline function Internal_Product(a::EFGFunction, b::SingleEFGMeasure)
    phi, dphi, ind, coord = b.phi, b.dphi, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return SingleEFGMeasure(as * phi, dphi, ind, coord)
end
@inline function Internal_Product(a::VecEFGFunction, b::SingleEFGMeasure)
    phi, ind, coord = b.phi, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return VecSingleEFGMeasure(as * phi, ind, coord)
end
@inline function Internal_Product(a::EFGFunction, b::VecSingleEFGMeasure)
    vec, ind, coord = b.vec, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return VecSingleEFGMeasure(as * vec, ind, coord)
end
@inline function Internal_Product(a::VecEFGFunction, b::VecSingleEFGMeasure)
    vec, ind, coord = b.vec, b.ind, b.coord
    as = Get_Point_Value(a, ind)
    return SingleEFGMeasure(dot(as, b.vec), vec, ind, coord)
end

# Functions-SingleMeasures
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

# Compositions of EFGFunctions with Functions

struct Composition{T<:Union{EFGFunction,VecEFGFunction}}
    f::Function
    Th::T
end

import Base: ∘
# Function ∘ EFGFunction → Composition
(∘)(a::Function, b::EFGFunction) = Composition{EFGFunction}(a, b)
(∘)(a::EFGFunction, b::Function) = Composition{EFGFunction}(b, a)
# Function ∘ VecEFGFunction → Composition
(∘)(a::Function, b::VecEFGFunction) = Composition{VecEFGFunction}(a, b)
(∘)(a::VecEFGFunction, b::Function) = Composition{VecEFGFunction}(b, a)

# Compositions and SingleEFGMeasures
@inline function Internal_Product(k::Composition{EFGFunction}, b::SingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)  # obtiene T en ese punto de Gauss
    return SingleEFGMeasure(val * b.phi, b.dphi, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{EFGFunction}, b::VecSingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)
    return VecSingleEFGMeasure(val * b.vec, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{VecEFGFunction}, b::SingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)  # obtiene T en ese punto de Gauss
    return SingleEFGMeasure(val * b.phi, b.dphi, b.ind, b.coord)
end

@inline function Internal_Product(k::Composition{VecEFGFunction}, b::VecSingleEFGMeasure)
    val = Get_Point_Value(k, b.ind)
    return VecSingleEFGMeasure(val * b.vec, b.ind, b.coord)
end
# --------------------------------------------------
# Evaluating Compositions at all Gauss Points
# --------------------------------------------------
@inline function Get_Point_Values(c::Composition{EFGFunction})
    vals = Get_Point_Values(c.Th)
    return c.f.(vals)
end

@inline function Get_Point_Values(c::Composition{VecEFGFunction})
    vals = Get_Point_Values(c.Th)
    return [c.f(v) for v in vals]  # aplica f a cada vector
end

# --------------------------------------------------
# Evaluating Compositions at a given Gauss Point
# --------------------------------------------------
@inline function Get_Point_Value(c::Composition{EFGFunction}, ind::Int)
    val = Get_Point_Value(c.Th, ind)
    return c.f(val)
end
@inline function Get_Point_Value(c::Composition{VecEFGFunction}, ind::Int)
    val = Get_Point_Value(c.Th, ind)
    return c.f(val)
end

#Constructor VectorField
struct VectorField{D,T}
    data::NTuple{D,T}
end

VectorField(x::Vararg{T,D}) where {T,D} = VectorField{D,T}(x)
Base.getindex(v::VectorField{D,T}, i::Int) where {D,T} = v.data[i]
Base.length(::VectorField{D,T}) where {D,T} = D

@inline function (∇::Nabla)(v::VectorField{D,SingleEFGMeasure}) where {D}
    return VectorField{D,VecSingleEFGMeasure}(
        ntuple(i -> ∇(v[i]), D)
    )
end

struct DivergenceScalar{D}
    divfield::VecSingleEFGMeasure
end

@inline divergence(v::VectorField{D,SingleEFGMeasure}) where {D} =
    DivergenceScalar{D}(VecSingleEFGMeasure(
        v[1].dphi,
        v[1].ind,
        v[1].coord))

@inline function Internal_Product(a::DivergenceScalar{D},
    b::DivergenceScalar{D}) where {D}
    M = zeros(Float64, D, D)
    ga = a.divfield.vec
    gb = b.divfield.vec

    @inbounds for i in 1:D
        for j in 1:D
            # Esto acopla el DoF i con el DoF j
            M[i, j] = ga[i] * gb[j]
        end
    end
    return M
end

struct AdjointGrad{D}
    g::VectorField{D,VecSingleEFGMeasure}
end

@inline adjoint(g::VectorField{D,VecSingleEFGMeasure}) where {D} =
    AdjointGrad{D}(g)

@inline field_dim(::Type{Float64}) = 1
@inline field_dim(::Type{VectorField{D,T}}) where {D,T} = D

@inline function Internal_Product(a::VectorField{D,SingleEFGMeasure},
    b::VectorField{D,SingleEFGMeasure}) where {D}
    M = zeros(Float64, D, D)
    @inbounds for α in 1:D
        M[α, α] = a[α].phi * b[α].phi
    end
    return M
end

@inline function Internal_Product(a::VectorField{D,VecSingleEFGMeasure},
    b::VectorField{D,VecSingleEFGMeasure}) where {D}
    M = zeros(Float64, D, D)
    @inbounds for α in 1:D
        M[α, α] = dot(a[α].vec, b[α].vec)
    end
    return M
end

@inline function Internal_Product(a::VectorField{D,VecSingleEFGMeasure},
    b::AdjointGrad{D}) where {D}
    M = zeros(Float64, D, D)
    @inbounds for i in 1:D
        @simd for j in 1:D
            M[i, j] = a[i].vec[j] * b.g[j].vec[i]
        end
    end
    return M
end

struct SumGrad{D}
    g::VectorField{D,VecSingleEFGMeasure}
    ag::AdjointGrad{D}
end
@inline function sum_tensors(g::VectorField{D,VecSingleEFGMeasure}, ag::AdjointGrad{D}) where {D}
    return SumGrad{D}(g, ag)
end

@inline function Internal_Product(a::VectorField{D,VecSingleEFGMeasure},
    b::SumGrad{D}) where {D}

    K_diag = Internal_Product(a, b.g)
    K_full = Internal_Product(a, b.ag)
    return K_diag + K_full
end

# Constructor para Integral
struct Integrand{T}
    object::T
end
const ∫ = Integrand

#Integrando un campo dado
@inline function Integrate(a::Integrand{<:AbstractVector}, b::DomainMeasure)
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a.object .* (jac .* weight)
end

struct Integrated
    arg::Any
    b::Union{DomainMeasure,AbstractVector{DomainMeasure}}
end

#Local Valuation of Integrals for assembling matrices and vectors
@inline function Integrate(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}})
    measures = isa(b, DomainMeasure) ? [b] : b
    return Integrated(a.object, measures)
end

struct MultiIntegrated
    terms::Vector{Integrated}
end

const AnyEFGField = Union{EFGFunction, VecEFGFunction, TransposedVecEFGFunction, VolumetricEFGFunction, CompositeEFGFunction}

import Base: +
import Base: -
import Base: *
import Base: ⋅
import Base: ⊙

@inline (+)(a::VectorField{D,T}, b::AdjointGrad{D}) where {D,T} = sum_tensors(a, b)
@inline (+)(a::Integrated, b::Integrated) = MultiIntegrated([a, b])
@inline (+)(a::MultiIntegrated, b::Integrated) = MultiIntegrated(vcat(a.terms, b))
@inline (+)(a::AnyEFGField, b::AnyEFGField) = CompositeEFGFunction([a, b], [1.0, 1.0])
@inline (+)(a::AnyEFGField, b::Any) = CompositeEFGFunction([a, b], [1.0, 1.0])
@inline (+)(a::Any, b::AnyEFGField) = CompositeEFGFunction([a, b], [1.0, 1.0])


@inline ()(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}}) = Integrate(a, b)
@inline ()(a::VecSingleEFGMeasure, b::VecSingleEFGMeasure) = a.vec * b.vec'

@inline (*)(a::Integrand, b::Union{DomainMeasure,AbstractVector{DomainMeasure}}) = Integrate(a, b)
@inline function (*)(a::Union{Float64,Int}, f::EFGFunction)
    new_fdom = [a * val for val in f.fdom]
    return EFGFunction(new_fdom, f.PHI, f.DPHI, f.D, f.Measure)
end
@inline (*)(f::EFGFunction, a::Union{Float64,Int}) = a * f
@inline function (*)(a::Union{Float64,Int}, f::VecEFGFunction)
    new_fdom = [a * val for val in f.fdom]
    return VecEFGFunction(new_fdom, f.VEC, f.D, f.Measure)
end
@inline (*)(f::VecEFGFunction, a::Union{Float64,Int}) = a * f
@inline (*)(a::Union{Float64,Int}, b::SingleEFGMeasure) = SingleEFGMeasure(a * b.phi, b.dphi, b.ind, b.coord)
@inline (*)(a::SingleEFGMeasure, b::Union{Float64,Int}) = SingleEFGMeasure(b * a.phi, a.dphi, a.ind, a.coord)
@inline (*)(a::Union{Float64,Int}, b::VecSingleEFGMeasure) = VecSingleEFGMeasure(a * b.vec, b.ind, b.coord)
@inline (*)(a::VecSingleEFGMeasure, b::Union{Float64,Int}) = VecSingleEFGMeasure(b * a.vec, a.ind, a.coord)
@inline (*)(a::VecSingleEFGMeasure, b::VecSingleEFGMeasure) = a.vec * b.vec'
@inline (*)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
@inline (*)(a::SingleEFGMeasure, b::SingleEFGMeasure) = a.phi * b.phi
@inline (*)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::EFGFunction, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::VecEFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::Function, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::Function, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline function (*)(f::Function, g::EFGFunction)
    gs = g.Measure.gs
    ngauss = length(g.PHI)
    d = size(gs, 2) - 2
    coord_1 = ntuple(j -> gs[1, j], d)
    val_1 = f(coord_1)
    if val_1 isa Number
        # --- Caso Escalar: devuelve EFGFunction ---
        new_PHI = Vector{Vector{Float64}}(undef, ngauss)
        for i in 1:ngauss
            c = ntuple(j -> gs[i, j], d)
            new_PHI[i] = g.PHI[i] * f(c)
        end
        return EFGFunction(g.fdom, new_PHI, g.DPHI, g.D, g.Measure)
    elseif val_1 isa VectorField || val_1 isa AbstractVector
        new_VEC = Vector{Matrix{Float64}}(undef, ngauss)
        for i in 1:ngauss
            c = ntuple(j -> gs[i, j], d)
            v_eval = f(c)
            v_vec = v_eval isa VectorField ? [v_eval[j] for j in 1:d] : v_eval
            new_VEC[i] = g.PHI[i] * v_vec'
        end
        return VecEFGFunction(g.fdom, new_VEC, g.D, g.Measure)
    end
end
@inline (*)(a::Composition{EFGFunction}, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::Composition{EFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::Composition{VecEFGFunction}, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::Composition{VecEFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (*)(a::DivergenceScalar{D}, b::DivergenceScalar{D}) where {D} = Internal_Product(a, b)
@inline (*)(a::Union{Float64,Int}, v::VectorField{D,SingleEFGMeasure}) where {D} = VectorField(ntuple(i -> a * v[i], D)...)
@inline (*)(v::VectorField{D,SingleEFGMeasure}, a::Union{Float64,Int}) where {D} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(a::VectorField{D,SingleEFGMeasure}, b::VectorField{D,SingleEFGMeasure}) where {D} = Internal_Product(a, b)
@inline (*)(v::VectorField{D,T}, a::Number) where {D,T<:Number} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(a::Number, v::VectorField{D,T}) where {D,T<:Number} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(f::Function, α::Number) = x -> f(x) * α
@inline (*)(α::Number, f::Function) = x -> α * f(x)
@inline (*)(k::Real, f::AnyEFGField) = CompositeEFGFunction([f], [Float64(k)])
@inline (*)(f::AnyEFGField, k::Real) = k * f

@inline (⋅)(a::Function, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::Function, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline function (⋅)(v::Function, g::VecEFGFunction)
    gs = g.Measure.gs
    ngauss = length(g.VEC)
    d = size(gs, 2) - 2 
    new_PHI = Vector{Vector{Float64}}(undef, ngauss)
    for i in 1:ngauss
        coord = ntuple(j -> gs[i, j], d)
        v_eval = v(coord)
        v_vec = v_eval isa VectorField ? [v_eval[c] for c in 1:d] : v_eval
        new_PHI[i] = g.VEC[i] * v_vec 
    end
    return EFGFunction(g.fdom, new_PHI, g.VEC, g.D, g.Measure)
end
@inline (⋅)(a::EFGFunction, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::VecEFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::EFGFunction, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
@inline (⋅)(a::VecEFGFunction, b::VecEFGFunction) = Internal_Product(a, b)
@inline (⋅)(a::SingleEFGMeasure, b::SingleEFGMeasure) = a.phi * b.phi
@inline (⋅)(a::VecSingleEFGMeasure, b::VecSingleEFGMeasure) = dot(a.vec, b.vec)
@inline (⋅)(a::Composition{EFGFunction}, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::Composition{EFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::Composition{VecEFGFunction}, b::SingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(a::Composition{VecEFGFunction}, b::VecSingleEFGMeasure) = Internal_Product(a, b)
@inline (⋅)(v::VectorField{D,T}, g::VecSingleEFGMeasure) where {D,T} =
    SingleEFGMeasure(sum(v[i] * g.vec[i] for i in 1:D), g.vec, g.ind, g.coord)
@inline (⋅)(a::VectorField{D,SingleEFGMeasure}, b::VectorField{D,SingleEFGMeasure}) where {D} = Internal_Product(a, b)
@inline (⋅)(a::Int, b::Int) = a * b
@inline (⋅)(a::Int, b::Vector{Float64}) = a * b
@inline (⋅)(a::Vector{Float64}, b::Int) = a * b
@inline (⋅)(v::VectorField{D,SingleEFGMeasure}, f::VectorField{D,T}) where {D,T<:Number} = VectorField(ntuple(i -> v[i] * f[i], D)...)
@inline (⋅)(f::VectorField{D,T}, v::VectorField{D,SingleEFGMeasure}) where {D,T<:Number} = VectorField(ntuple(i -> v[i] * f[i], D)...)
@inline (⋅)(::Nabla, v::VectorField{D,SingleEFGMeasure}) where {D} = divergence(v)
@inline function (⋅)(f::Function, v::VectorField{D,SingleEFGMeasure}) where D
    coord = v[1].coord
    val_f = f(coord)
    return VectorField(ntuple(i -> v[i] * val_f[i], D)...)
end

@inline (⋅)(v::VectorField{D,SingleEFGMeasure}, f::Function) where D = f ⋅ v
@inline (⋅)(::Nabla, f::EFGFunction) = VolumetricEFGFunction(f)
#(⋅)(::typeof(∇), v::VectorField{D,SingleEFGMeasure}) where {D} = divergence(v)
#(⋅)(∇::Nabla, v::VectorField{D,SingleEFGMeasure}) where {D} = divergence(v)

@inline (⊙)(a::VectorField{D,VecSingleEFGMeasure}, b::VectorField{D,VecSingleEFGMeasure}) where {D} = Internal_Product(a, b)
@inline (⊙)(a::VectorField{D,VecSingleEFGMeasure}, b::AdjointGrad{D}) where {D} = Internal_Product(a, b)
@inline (⊙)(a::VectorField{D,VecSingleEFGMeasure}, b::SumGrad{D}) where {D} = Internal_Product(a, b)

# Nothing Operations
@inline (+)(::Nothing, ::Nothing) = nothing
@inline (*)(::Nothing, ::Nothing) = nothing
@inline (⋅)(::Nothing, ::Nothing) = nothing
@inline (*)(::Any, ::Nothing) = nothing
@inline (⋅)(::Any, ::Nothing) = nothing
@inline (⋅)(::Nothing, ::VectorField{D,T}) where {D,T<:Number} = nothing
@inline (⋅)(::VectorField{D,T}, ::Nothing,) where {D,T<:Number} = nothing
@inline adjoint(::Nothing) = nothing
@inline (-)(a::Nothing, b::Nothing) = nothing
@inline ⊙(::Nothing, ::Nothing) = nothing
@inline (⋅)(::Nothing, ::Function) = nothing
@inline (⋅)(::Function, ::Nothing) = nothing
@inline (*)(::Nothing, ::Function) = nothing
@inline (*)(::Function, ::Nothing) = nothing