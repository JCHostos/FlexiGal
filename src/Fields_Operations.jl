import Base: +
import Base: -
import Base: *
import LinearAlgebra: ⋅ , tr, I

function ⊙ end

struct EFGCache{T}
    last_ind::Base.RefValue{Int32}
    last_val::Base.RefValue{T}
end

# Constructor para inicializar vacío
function EFGCache(::Type{T}) where T
    # Inicializamos con -1 para que la primera comparación siempre falle
    return EFGCache{T}(Ref(Int32(-1)), Ref(zero(T)))
end

struct EFGFunction{D, L, G}
    field_nodal::Vector{Float64}
    PHI::Vector{Vector{Float64}}
    DPHI::Vector{Vector{SVector{G, Float64}}}
    DOM::Vector{Vector{Int}}
    D::Int
    Measure::DomainMeasure
    coeff::Float64
    cache::EFGCache{L}
end

@inline function EFGFunction(field_nodal::Vector{Float64}, Space::EFGSpace, Measure::DomainMeasure)
    Shapes = EFG_Measure(Measure, Space) 
    D = field_dim(Space.Field_Type)
    DG = typeof(Shapes).parameters[1] 
    L_type = D == 1 ? Float64 : SVector{D, Float64}
    return EFGFunction{D, L_type, DG}(field_nodal,Shapes.PHI,Shapes.DPHI,Shapes.DOM,D,Measure,1.0,EFGCache(L_type))
end

@inline Get_Nodal_Values(a::EFGFunction{D, L, G}) where {D, L, G} = a.field_nodal

@inline function Get_Point_Values(f::EFGFunction{D, L, G}) where {D, L, G}
    ngauss = length(f.PHI)
    res = Vector{L}(undef, ngauss)
    field_nodal = f.field_nodal
    DOM = f.DOM
    PHI = f.PHI
    coeff = f.coeff
    @inbounds for i in 1:ngauss
        local_nodes = DOM[i]
        phi = PHI[i]
        acc = zero(L)
        for a in eachindex(local_nodes)
            gn = local_nodes[a]
            val_node = if D == 1
                field_nodal[gn]
            else
                idx_base = (gn - 1) * D
                L(ntuple(d -> field_nodal[idx_base + d], D))
            end
            acc += phi[a] * val_node
        end
        res[i] = acc * coeff
    end
    return res
end

struct VecEFGFunction{D, L, G, M, T_RES, F<:EFGFunction} 
    source::F
    coeff::Float64
    cache::EFGCache{T_RES}
end

@inline Get_Nodal_Values(a::VecEFGFunction{D, L, G, M, T_RES, F}) where {D, L, G, M, T_RES, F} = a.source.field_nodal

struct Nabla end
const ∇ = Nabla()

@inline function (∇::Nabla)(f::EFGFunction{D, L, G}) where {D, L, G}
    T_RES = D == 1 ? SVector{G, Float64} : SMatrix{D, G, Float64, D * G}
    return VecEFGFunction{D, L, G, :grad, T_RES, typeof(f)}(f, f.coeff, EFGCache(T_RES))
end

@inline function Get_Point_Values(f::VecEFGFunction{D, L, G, :grad, T_RES}) where {D, L, G, T_RES}
    source = f.source
    ngauss = length(source.DPHI)
    res = Vector{T_RES}(undef, ngauss)
    fn = source.field_nodal
    DOM = source.DOM
    VEC_all = source.DPHI
    coeff = f.coeff
    @inbounds for i in 1:ngauss
        nodes = DOM[i]
        dphi = VEC_all[i]
        acc = zero(T_RES)
        for a in eachindex(nodes)
            gn = nodes[a]
            val_node = if D == 1
                fn[gn]
            else
                idx_base = (gn - 1) * D
                L(ntuple(d -> fn[idx_base + d], Val(D))) # Constructor L respetado
            end
            term = if D == 1
                dphi[a] * val_node
            else
                val_node * dphi[a]'
            end
            acc += term
        end
        res[i] = acc * coeff
    end
    return res
end

struct TransposedVecEFGFunction{D, L, G, T_RES, F}
    origin::VecEFGFunction{D, L, G, :grad, T_RES, F}
    D::Int
end

@inline Get_Nodal_Values(a::TransposedVecEFGFunction{D, L, G, T_RES, F}) where {D, L, G, T_RES, F} = Get_Nodal_Values(a.origin)

struct CompositeEFGFunction{D, L, T1, T2}
    f1::T1
    f2::T2
    c1::Float64
    c2::Float64
    D::Int
end

import LinearAlgebra: adjoint

# Sobrecarga del operador ' (adjunto)
@inline adjoint(f::VecEFGFunction{D, L, G, :grad, T_RES, F}) where {D, L, G, T_RES, F} = 
    TransposedVecEFGFunction{D, L, G, T_RES, F}(f, f.source.D)

@inline function Get_Point_Values(f::TransposedVecEFGFunction)
    vals_orig = Get_Point_Values(f.origin) 
    return [v' for v in vals_orig]
end

struct VolumetricEFGFunction{D, L, G, T_ORIG}
    origin::T_ORIG
    D::Int
end

@inline Get_Nodal_Values(a::VolumetricEFGFunction{D, L, G, T_ORIG}) where {D, L, G, T_ORIG} = Get_Nodal_Values(a.origin)


@inline function VolumetricEFGFunction(f::T, D::Int) where {T}
    params = typeof(f).parameters
    return VolumetricEFGFunction{params[1], params[2], params[3], T}(f, D)
end

@inline function (⋅)(::Nabla, f::EFGFunction{D,L,G}) where {D,L,G}
    g = ∇(f)
    return VolumetricEFGFunction(g, f.D)
end

@inline function Get_Point_Values(f::VolumetricEFGFunction{D,L,G,T_ORIG}) where {D,L,G,T_ORIG <: VecEFGFunction}
    f_orig = f.origin 
    source = f_orig.source
    fn = source.field_nodal
    DOM = source.DOM
    VEC_all = source.DPHI
    ngauss = length(VEC_all)
    res = Vector{Float64}(undef, ngauss)
    @inbounds for i in 1:ngauss
        nodes = DOM[i]
        base_vecs = VEC_all[i]
        div_val = 0.0
        for a in eachindex(nodes)
            gn = nodes[a]
            val_node = D == 1 ? fn[gn] : L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
            div_val += dot(val_node, base_vecs[a])
        end
        res[i] = div_val * f_orig.coeff 
    end
    return res
end

@inline function Get_Point_Values(f::VolumetricEFGFunction{D,L,G,T_ORIG}) where {D,L,G,T_ORIG}
    vals_orig = Get_Point_Values(f.origin)
    return tr.(vals_orig)
end

@inline function Get_Point_Values(f::VecEFGFunction{D,L,G,:vol,T_RES}) where {D,L,G,T_RES}
    source = f.source # REDIRECCIÓN
    fn = source.field_nodal
    DOM = source.DOM
    VEC_all = source.DPHI
    
    ngauss = length(VEC_all)
    res = Vector{T_RES}(undef, ngauss)
    Id_mat = one(T_RES) 
    @inbounds for i in 1:ngauss
        nodes = DOM[i]
        base_vecs = VEC_all[i]
        div_val = 0.0
        for a in eachindex(nodes)
            gn = nodes[a]
            val_node = D == 1 ? fn[gn] : L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
            div_val += dot(val_node, base_vecs[a])
        end
        res[i] = (div_val * f.coeff) * Id_mat
    end
    return res
end

@inline function Get_Point_Values(f::CompositeEFGFunction{D,L}) where {D,L}
    vals1 = Get_Point_Values(f.f1)
    vals2 = Get_Point_Values(f.f2)
    return f.c1 .* vals1 .+ f.c2 .* vals2
end

struct ScalarFuncXEFGFunction{D,L,G,F,T,DG,L_OUT}
    f::F        
    origin::T
end

struct VectorFuncXEFGFunction{D,L,G,F,T,DG,L_OUT}
    v::F         
    origin::T
end

@inline function (*)(f::F, g::EFGFunction{D, L, G}) where {F<:Function, D, L, G}
    raw_cols = size(g.Measure.gs, 2)
    DG = raw_cols - 2 
    x_test = SVector{DG, Float64}(ntuple(j -> g.Measure.gs[1, j], Val(DG)))
    val_test = f(x_test)
    if val_test isa Number
        L_OUT = L
        return ScalarFuncXEFGFunction{D, L, G, F, typeof(g), DG, L_OUT}(f, g)
    else
        V_DIM = length(val_test)
        L_OUT = (L == Float64) ? SVector{V_DIM, Float64} : Float64
        return VectorFuncXEFGFunction{D, L, G, F, typeof(g), DG, L_OUT}(f, g)
    end
end

@inline function Get_Point_Values(field::ScalarFuncXEFGFunction{D,L,G,F,T,DG,L_OUT}) where {D,L,G,F,T,DG,L_OUT}
    g = field.origin
    gs = g.Measure.gs
    ngauss = length(g.PHI)
    res = Vector{L_OUT}(undef, ngauss)
    fn = g.field_nodal
    DOM = g.DOM
    PHI = g.PHI
    c = g.coeff 
    @inbounds for i in 1:ngauss
        x = SVector{DG, Float64}(ntuple(j -> gs[i, j], Val(DG)))
        nodes = DOM[i]
        phi = PHI[i]
        acc_g = zero(L)
        for a in eachindex(nodes)
            gn = nodes[a]
            val_node = if D == 1
                fn[gn]
            else
                L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
            end
            acc_g += phi[a] * val_node
        end
        res[i] = field.f(x) * (acc_g * c)
    end
    return res
end

@inline function Get_Point_Values(field::VectorFuncXEFGFunction{D,L,G,F,T,DG,L_OUT}) where {D,L,G,F,T,DG,L_OUT}
    g = field.origin
    gs = g.Measure.gs
    ngauss = length(g.PHI)
    res = Vector{L_OUT}(undef, ngauss)
    fn = g.field_nodal
    DOM = g.DOM
    PHI = g.PHI
    c = g.coeff
    @inbounds for i in 1:ngauss
        x = SVector{DG, Float64}(ntuple(j -> gs[i, j], Val(DG)))
        nodes = DOM[i]
        phi = PHI[i]
        acc_g = zero(L)
        for a in eachindex(nodes)
            gn = nodes[a]
            val_node = if D == 1
                fn[gn]
            else
                L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
            end
            acc_g += phi[a] * val_node
        end
        val_final = acc_g * c
        v_x = field.v(x)
        if L == Float64
            res[i] = v_x * val_final
        else
            res[i] = dot(v_x, val_final)
        end
    end
    return res
end

@inline (∇::Nabla)(::Nothing) = nothing
@inline (∇::Nabla)(::Number) = 0.0
# Operations over EFGMeasures
struct SingleEFGMeasure{DG}
    phi::Float64
    dphi::SVector{DG, Float64}
    coord::SVector{DG, Float64}
    ind::Int
end

struct VecSingleEFGMeasure{DG}
    vec::SVector{DG, Float64}
    coord::SVector{DG, Float64}
    ind::Int
end

@inline (-)(m::SingleEFGMeasure{DG}) where DG = SingleEFGMeasure{DG}(-m.phi, -m.dphi, m.coord, m.ind)

@inline (-)(m::VecSingleEFGMeasure{DG}) where DG = VecSingleEFGMeasure{DG}(-m.vec, m.coord, m.ind)

@inline function (∇::Nabla)(s::SingleEFGMeasure{DG}) where DG
    return VecSingleEFGMeasure{DG}(s.dphi, s.coord,s.ind)
end

struct DivergenceScalar{D,DG}
    divfield::VecSingleEFGMeasure{DG}
end

struct TransformedMeasure{D, DG, T_MAT, T_MEAS}
    mat::T_MAT
    m::T_MEAS  
end

struct AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}
    mat::T_MAT
    m::T_MEAS
end
const AnyEFGMeasure{DG} = Union{SingleEFGMeasure{DG}, VecSingleEFGMeasure{DG},DivergenceScalar}

@inline function evaluate_at_point(f::EFGFunction{D, L, G}, b::AnyEFGMeasure) where {D, L, G}
    idx = b.ind
    if f.cache.last_ind[] == idx
        return f.cache.last_val[]
    end
    nodes = f.DOM[idx]
    phi = f.PHI[idx]
    fn = f.field_nodal
    acc = zero(L)
    @inbounds for a in eachindex(nodes)
        gn = nodes[a]
        val_node = D == 1 ? fn[gn] : L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
        acc += phi[a] * val_node
    end
    res = acc * f.coeff
    f.cache.last_ind[] = idx
    f.cache.last_val[] = res
    return res
end

@inline function evaluate_at_point(f::VecEFGFunction{D, L, G, :grad, T_RES}, b::AnyEFGMeasure) where {D, L, G, T_RES}
    idx = b.ind
    if f.cache.last_ind[] == idx
        return f.cache.last_val[]
    end
    source = f.source
    nodes = source.DOM[idx]
    dphi = source.DPHI[idx]
    fn = source.field_nodal
    acc = zero(T_RES)
    @inbounds for a in eachindex(nodes)
        gn = nodes[a]
        val_node = D == 1 ? fn[gn] : L(ntuple(d -> fn[(gn-1)*D + d], Val(D)))
        term = D == 1 ? dphi[a] * val_node : val_node * dphi[a]'
        acc += term
    end
    res = acc * f.coeff
    f.cache.last_ind[] = idx
    f.cache.last_val[] = res
    return res
end

@inline function evaluate_at_point(obj::ScalarFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}, b::AnyEFGMeasure) where 
    {D, L, G, F, T, DG, L_OUT}
    val_origin = evaluate_at_point(obj.origin, b) 
    return obj.f(b.coord)::L_OUT * val_origin
end

@inline function evaluate_at_point(obj::VectorFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}, b::AnyEFGMeasure) where 
    {D, L, G, F, T, DG, L_OUT}
    val_origin = evaluate_at_point(obj.origin, b)
    return obj.v(b.coord)::L_OUT * val_origin
end

@inline function evaluate_at_point(f::VolumetricEFGFunction{D, L, G, T_ORIG}, b::AnyEFGMeasure) where {D, L, G, T_ORIG}
    val_grad = evaluate_at_point(f.origin, b)
    return tr(val_grad)
end

@inline function evaluate_at_point(f::TransposedVecEFGFunction, b::AnyEFGMeasure)
    return evaluate_at_point(f.origin, b)'
end

@inline function evaluate_at_point(f::CompositeEFGFunction{D, L, T1, T2}, b::AnyEFGMeasure) where {D, L, T1, T2}
    val1 = evaluate_at_point(f.f1, b)
    val2 = evaluate_at_point(f.f2, b) 
    return f.c1 * val1 + f.c2 * val2
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

# Functions-SingleMeasures
@inline function Internal_Product(a::Function, b::SingleEFGMeasure{DG}) where DG
    as = a(b.coord)
    if as isa Number
        return SingleEFGMeasure{DG}(as * b.phi, b.dphi, b.coord, b.ind)
    elseif as isa SVector || as isa VectorField
        # Si la función devuelve un vector, la multiplicamos por el escalar phi
        return VecSingleEFGMeasure{DG}(SVector{DG,Float64}(as) * b.phi, b.coord, b.ind)
    else
        error("Internal_Product: el valor de retorno de la función no es soportado.")
    end
end

@inline function Internal_Product(a::Function, b::VecSingleEFGMeasure{DG}) where DG
    as = a(b.coord) 
    if as isa Number
        return VecSingleEFGMeasure{DG}(as * b.vec, b.coord, b.ind)
    elseif as isa VectorField
        val_svector = SVector{DG, Float64}(as.data)
        value = dot(val_svector, b.vec)
        return SingleEFGMeasure{DG}(value, b.vec, b.coord, b.ind)
    else
        value = dot(as, b.vec)
        return SingleEFGMeasure{DG}(value, b.vec, b.coord, b.ind)
    end
end

struct ProductEFGFunction{D, L, G, T1, T2}
    f1::T1
    f2::T2
    D::Int
    coeff::Float64
end

# Ajustamos la firma para que acepte el parámetro F de los wrappers
@inline function (*)(f1::TransposedVecEFGFunction{D,L,G,T_RES1,F1}, 
    f2::VecEFGFunction{D,L,G,M,T_RES2,F2}) where {D,L,G,T_RES1,F1,M,T_RES2,F2} 
    return ProductEFGFunction{D, L, G, typeof(f1), typeof(f2)}(f1, f2, D, 1.0)
end

@inline function Get_Point_Values(f::ProductEFGFunction{D, L, G}) where {D, L, G}
    vals1 = Get_Point_Values(f.f1) 
    vals2 = Get_Point_Values(f.f2)
    return (f.coeff) .* (vals1 .* vals2)
end

@inline function evaluate_at_point(f::ProductEFGFunction, b::AnyEFGMeasure)
    m1 = evaluate_at_point(f.f1, b)
    m2 = evaluate_at_point(f.f2, b)
    return f.coeff * (m1 * m2)
end

@inline function Get_Point_Values(f::VolumetricEFGFunction{D, L, G, <:ProductEFGFunction}) where {D, L, G}
    vals_prod = Get_Point_Values(f.origin) 
       return tr.(vals_prod)
end

@inline function tr(f::VecEFGFunction{D, L, G, :grad, T_RES, F}) where {D, L, G, T_RES, F}
    return VolumetricEFGFunction(f, f.source.D)
end

@inline function tr(f::VecEFGFunction{D, L, G, :vol, T_RES, F}) where {D, L, G, T_RES, F}
    return VolumetricEFGFunction(f, f.source.D)
end

@inline tr(f::ProductEFGFunction) = VolumetricEFGFunction(f, f.D)

@inline tr(f::TransposedVecEFGFunction) = tr(f.origin)

@inline tr(f::CompositeEFGFunction) = f.c1 * tr(f.f1) + f.c2 * tr(f.f2)


#Constructor VectorField
struct VectorField{D,T}
    data::NTuple{D,T}
end

VectorField(x::Vararg{T,D}) where {T,D} = VectorField{D,T}(x)
Base.getindex(v::VectorField{D,T}, i::Int) where {D,T} = v.data[i]
Base.length(::VectorField{D,T}) where {D,T} = D

# Transform VectorField into a SVector without cost
@inline Base.convert(::Type{SVector{D,T}}, v::VectorField{D,T}) where {D,T} = SVector{D,T}(v.data)
@inline SVector{D,T}(v::VectorField{D,T}) where {D,T} = SVector{D,T}(v.data)

# Allow VectorField to behave as an iterable for ntuple and others
@inline Base.Tuple(v::VectorField) = v.data

@inline (-)(v::VectorField{D, T}) where {D, T} = VectorField{D, T}(ntuple(i -> -v.data[i], Val(D)))

@inline function (∇::Nabla)(v::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG}
    return VectorField{D,VecSingleEFGMeasure{DG}}(ntuple(i -> ∇(v[i]), Val(D)))
end

@inline function divergence(v::VectorField{D, SingleEFGMeasure{DG}}) where {D,DG}
    return DivergenceScalar{D,DG}(VecSingleEFGMeasure{DG}(v[1].dphi, v[1].coord,v[1].ind))
end

@inline function Internal_Product(a::DivergenceScalar{D,DG},
                                  b::DivergenceScalar{D,DG}) where {D,DG}
    ga = a.divfield.vec
    gb = b.divfield.vec
    return ga * gb'
end

struct AdjointGrad{D,DG}
    g::VectorField{D,VecSingleEFGMeasure{DG}}
end

@inline adjoint(g::VectorField{D,VecSingleEFGMeasure{DG}}) where {D,DG} =
    AdjointGrad{D,DG}(g)

@inline field_dim(::Type{Float64}) = 1
@inline field_dim(::Type{VectorField{D,T}}) where {D,T} = D

@inline function Internal_Product(a::VectorField{D,SingleEFGMeasure{DG}}, 
                                 b::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG}
    vals = ntuple(i -> a.data[i].phi * b.data[i].phi, Val(D))
    return SDiagonal{D, Float64}(vals)
end

@inline function Internal_Product(a::VectorField{D,VecSingleEFGMeasure{DG}},
                                 b::VectorField{D,VecSingleEFGMeasure{DG}}) where {D,DG}
    vals = ntuple(i -> dot(a.data[i].vec, b.data[i].vec), Val(D))
    return SDiagonal{D, Float64}(vals)
end

@inline function Internal_Product(a::VectorField{D,VecSingleEFGMeasure{DG}},
                                  b::AdjointGrad{D,DG}) where {D,DG}
    data_a = a.data
    data_b = b.g.data
    return SMatrix{D,D,Float64}(ntuple(k -> begin
        i = (k - 1) % D + 1
        j = (k - 1) ÷ D + 1
        return data_a[i].vec[j] * data_b[j].vec[i]
    end, Val(D * D)))
end

@inline function Internal_Product(a::AdjointGrad{D,DG}, 
                                  b::AdjointGrad{D,DG}) where {D,DG}
    return Internal_Product(a.g, b.g)
end

@inline function Base.getproperty(t::TransformedMeasure, s::Symbol)
    if s === :ind || s === :coord
        return getproperty(getfield(t, :m), s)
    else
        return getfield(t, s)
    end
end

@inline function _apply_transform(f_eval, vf::VectorField{D, T_MEAS}) where {D, T_MEAS}
    mat_val = evaluate_at_point(f_eval, vf.data[1])
    DG = get_dg(T_MEAS)
    T_RES = TransformedMeasure{D, DG, typeof(mat_val), T_MEAS}
    return VectorField{D, T_RES}(ntuple(i -> T_RES(mat_val, vf.data[i]), Val(D)))
end
# Agregamos los parámetros de tipo faltantes (T_RES, F, etc.) para que el despacho sea correcto
@inline (*)(f::VecEFGFunction{D,L,G,M,T_RES,F}, v::VectorField{DV, <:VecSingleEFGMeasure}) where {D,L,G,M,T_RES,F,DV} = _apply_transform(f, v)
@inline (*)(f::TransposedVecEFGFunction{D,L,G,T_RES,F}, v::VectorField{DV, <:VecSingleEFGMeasure}) where {D,L,G,T_RES,F,DV} = _apply_transform(f, v)
@inline function _apply_adjoint_transport(f_eval, vf_du::VectorField{D, T_MEAS}) where {D, T_MEAS}
    mat_val = evaluate_at_point(f_eval, vf_du.data[1])
    DG = get_dg(T_MEAS)
    T_RES = AdjointTransportMeasure{D, DG, typeof(mat_val), T_MEAS}
    return VectorField{D, T_RES}(ntuple(i -> T_RES(mat_val, vf_du.data[i]), Val(D)))
end
@inline (*)(f::CompositeEFGFunction, v::VectorField{D, <:VecSingleEFGMeasure}) where {D} = 
    (f.c1 * (f.f1 * v)) + (f.c2 * (f.f2 * v))
@inline (*)(f::CompositeEFGFunction, v::VectorField{D, <:TransformedMeasure}) where {D} = 
    (f.c1 * (f.f1 * v)) + (f.c2 * (f.f2 * v))
@inline function (*)(f::ProductEFGFunction, v::VectorField{D, <:VecSingleEFGMeasure}) where {D}
    return _apply_transform(f, v)
end
@inline function (*)(f::ProductEFGFunction, v::VectorField{D, <:TransformedMeasure}) where {D}
    return _apply_transform(f, v)
end

# Para las trazas (VolumetricEFGFunction)
@inline (*)(f::VolumetricEFGFunction, v::VectorField{D, <:VecSingleEFGMeasure}) where {D} = _apply_transform(f, v)
@inline (*)(f::VolumetricEFGFunction, v::VectorField{D, <:TransformedMeasure}) where {D} = _apply_transform(f, v)

# Para evitar que te pase lo mismo si usas un campo escalar puro (EFGFunction)
@inline (*)(f::EFGFunction, v::VectorField{D, <:VecSingleEFGMeasure}) where {D} = _apply_transform(f, v)
@inline (*)(f::EFGFunction, v::VectorField{D, <:TransformedMeasure}) where {D} = _apply_transform(f, v)

# Por si usas funciones espaciales escalares (ScalarFuncXEFGFunction)
@inline (*)(f::ScalarFuncXEFGFunction, v::VectorField{D, <:VecSingleEFGMeasure}) where {D} = _apply_transform(f, v)
@inline (*)(f::ScalarFuncXEFGFunction, v::VectorField{D, <:TransformedMeasure}) where {D} = _apply_transform(f, v)


struct LazySumGrads{D, DG, T<:Tuple}
    terms::T
end


struct IdentityOperator end
const Id = IdentityOperator()

@inline get_dg(::VectorField{D, T}) where {D, T} = get_dg(T)
@inline get_dg(::Type{VectorField{D, T}}) where {D, T} = get_dg(T)
@inline get_dg(::VecSingleEFGMeasure{DG}) where {DG} = DG
@inline get_dg(::Type{VecSingleEFGMeasure{DG}}) where {DG} = DG
@inline get_dg(::AdjointGrad{D, DG}) where {D, DG} = DG
@inline get_dg(::Type{AdjointGrad{D, DG}}) where {D, DG} = DG
@inline get_dg(::LazySumGrads{D, DG}) where {D, DG} = DG
@inline get_dg(::Type{LazySumGrads{D, DG}}) where {D, DG} = DG
@inline get_dg(::TransformedMeasure{D, DG}) where {D, DG} = DG
@inline get_dg(::Type{<:TransformedMeasure{D, DG}}) where {D, DG} = DG
@inline get_dg(::Type{DivergenceScalar{D, DG}}) where {D, DG} = DG
@inline get_dg(::Type{AdjointTransportMeasure{D, DG, T1, T2}}) where {D, DG, T1, T2} = DG

const AnyGradTerm{D, DG} = Union{
    VectorField{D, VecSingleEFGMeasure{DG}},
    VectorField{D, <:TransformedMeasure},
    VectorField{D, <:AdjointTransportMeasure},
    TransformedMeasure{D, DG, <:Any, <:DivergenceScalar},
    AdjointGrad{D, DG},
    LazySumGrads{D, DG}
}

@inline function tr(v::VectorField{D, VecSingleEFGMeasure{DG}}) where {D, DG}
    return DivergenceScalar{D, DG}(v.data[1])
end

@inline tr(ag::AdjointGrad) = tr(ag.g)

@inline function tr(v::VectorField{D, <:TransformedMeasure}) where {D}
    m = v.data[1].m      # Esto es ∇ϕ_J (SVector)
    mat = v.data[1].mat  # Esto es (∇uh)' (SMatrix)
    new_vec = mat' * m.vec  
    new_m = VecSingleEFGMeasure{get_dg(v)}(new_vec, m.coord, m.ind)
    return DivergenceScalar{D, get_dg(v)}(new_m)
end

@inline function tr(v::VectorField{D, <:AdjointTransportMeasure}) where {D}
    m = v.data[1].m 
    mat = v.data[1].mat
    new_vec = mat * m.vec 
    new_m = VecSingleEFGMeasure{get_dg(v)}(new_vec, m.coord, m.ind)
    return DivergenceScalar{D, get_dg(v)}(new_m)
end

@inline tr(d::DivergenceScalar) = d

@inline function tr(ls::LazySumGrads{D,DG}) where {D,DG}
    new_terms = map(tr, ls.terms)
    return LazySumGrads{D, DG, typeof(new_terms)}(new_terms)
end

@inline get_D(::DivergenceScalar{D, DG}) where {D, DG} = D
@inline get_DG(::DivergenceScalar{D, DG}) where {D, DG} = DG
@inline get_D(t::Tuple) = get_D(t[1])
@inline get_DG(t::Tuple) = get_DG(t[1])

@inline flatten_sum(a::LazySumGrads) = a.terms
@inline flatten_sum(a) = (a,)

@inline field_dim(::Type{LazySumGrads{D, DG, T}}) where {D, DG, T} = D
@inline field_dim(::Type{AdjointGrad{D, DG}}) where {D, DG} = D
@inline field_dim(::Type{<:DivergenceScalar{D}}) where D = D
@inline field_dim(::Type{<:TransformedMeasure{D}}) where D = D
@inline function (+)(a::T1, b::T2) where {T1<:AnyGradTerm, T2<:AnyGradTerm}
    D = field_dim(T1)
    DG = get_dg(a) 
    new_tuple = (flatten_sum(a)..., flatten_sum(b)...)
    return LazySumGrads{D, DG, typeof(new_tuple)}(new_tuple)
end

struct ScalarTimesId{T_SCALAR, D}
    f::T_SCALAR
end


@inline function (*)(d::DivergenceScalar{D, DG}, ::IdentityOperator) where {D, DG}
    return TransformedMeasure{D, DG, Float64, DivergenceScalar{D, DG}}(1.0, d)
end

@inline function (*)(d::DivergenceScalar{D, DG}, s::ScalarTimesId) where {D, DG}
    return TransformedMeasure{D, DG, typeof(s.f), DivergenceScalar{D, DG}}(s.f, d)
end


@inline function ScalarTimesId(f::T, D::Int) where T
    return ScalarTimesId{T, D}(f)
end

@inline function (*)(f::Union{VolumetricEFGFunction, CompositeEFGFunction}, ::IdentityOperator)
    return ScalarTimesId(f, f.D)
end

@inline function Get_Point_Values(f::ScalarTimesId{T, D}) where {T, D}
    vals_escalares = Get_Point_Values(f.f)
    Id_mat = SMatrix{D, D, Float64}(I) 
    return [v * Id_mat for v in vals_escalares]
end

@inline function evaluate_at_point(f::ScalarTimesId, b::AnyEFGMeasure)
    return evaluate_at_point(f.f, b) 
end

const AnyEFGField = Union{EFGFunction,VecEFGFunction,TransposedVecEFGFunction,VolumetricEFGFunction,
CompositeEFGFunction,ScalarFuncXEFGFunction,VectorFuncXEFGFunction,ProductEFGFunction,ScalarTimesId,IdentityOperator}

struct Composition{T<:AnyEFGField}
    f::Function
    field::T
end

import Base: ∘

(f::Function) ∘ (u::AnyEFGField) = Composition(f, u)
(u::AnyEFGField) ∘ (f::Function) = Composition(f, u)

@inline function evaluate_at_point(comp::Composition, measure::SingleEFGMeasure{DG}) where {DG}
    val_inner = evaluate_at_point(comp.field, measure)
    return comp.f(val_inner)
end

@inline function evaluate_at_point(comp::Composition, measure::VecSingleEFGMeasure{DG}) where {DG}
    val_inner = evaluate_at_point(comp.field, measure)
    return comp.f(val_inner)
end


@inline evaluate_at_point(f::AnyEFGField, d::DivergenceScalar) = evaluate_at_point(f, d.divfield)

@inline evaluate_at_point(::IdentityOperator, b::AnyEFGMeasure) = 1.0

@inline evaluate_at_point(::IdentityOperator, d::DivergenceScalar) = 1.0

# --- FUNCIONES AUXILIARES PARA EVITAR EL "ISA" ---
@inline _get_val(m::DivergenceScalar, i) = m.divfield.vec[i]
@inline _get_val(m, i) = m.vec[i]

@inline function Internal_Product(a::AdjointGrad{D, DG}, 
                                  b::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}) where {D, DG, T_MAT, T_MEAS}
    data_a, data_b = a.g.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(k -> begin
        i, j = (k - 1) % D + 1, (k - 1) ÷ D + 1
        term = 0.0
        @inbounds for m in 1:D
            term += data_a[i].vec[m] * data_b[j].mat[m, j]
        end
        m_val = _get_val(data_b[j].m, i)     
        return term * m_val
    end, Val(D * D)))
end

# 2. VEC FIELD (NORMAL) vs TRANSFORMED
@inline function Internal_Product(a::VectorField{D, VecSingleEFGMeasure{DG}}, 
                                  b::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}) where {D, DG, T_MAT, T_MEAS}
    data_a, data_b = a.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(k -> begin
        i, j = (k - 1) % D + 1, (k - 1) ÷ D + 1
        mat_val = if T_MAT <: Number
            (i == j) ? data_b[j].mat : 0.0
        else
            data_b[j].mat[i, j]
        end
        if mat_val == 0.0
            return 0.0
        end
        grad_I = data_a[i].vec
        grad_J = data_b[j].m.vec
        dot_grad = 0.0
        @inbounds for n in 1:D
            dot_grad += grad_I[n] * grad_J[n]
        end
        return mat_val * dot_grad
    end, Val(D * D)))
end

@inline function Internal_Product(a::AnyGradTerm{D, DG}, 
                                  b::TransformedMeasure{D, DG, T_MAT, <:DivergenceScalar}) where {D, DG, T_MAT}
    lambda_val = (T_MAT <: AbstractMatrix) ? b.mat[1,1] : b.mat
    return lambda_val * (tr(a) * b.m)
end

# 3. TRANSFORMED vs TRANSFORMED
@inline function Internal_Product(a::VectorField{D, TransformedMeasure{D, DG, T_MAT1, T_MEAS1}}, 
                                  b::VectorField{D, TransformedMeasure{D, DG, T_MAT2, T_MEAS2}}) where {D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2}
    data_a, data_b = a.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(k -> begin
        i, j = (k - 1) % D + 1, (k - 1) ÷ D + 1
        va = _get_val(data_a[i].m, i)
        vb = _get_val(data_b[j].m, j)
        common = 0.0
        @inbounds for m in 1:D, n in 1:D
            common += data_a[i].mat[m, n] * data_b[j].mat[m, n]
        end
        return common * va * vb
    end, Val(D * D)))
end

@inline function Internal_Product(a::VectorField{D, VecSingleEFGMeasure{DG}}, 
                                  b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}) where {D, DG, T_MAT, T_MEAS}
    data_a, data_b = a.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(k -> begin
        i, j = (k - 1) % D + 1, (k - 1) ÷ D + 1
        grad_I = data_a[i].vec     # ∇ϕ_I
        grad_J = data_b[j].m.vec   # ∇ϕ_J
        mat = data_b[j].mat        # ∇uh        
        val = 0.0
        @inbounds for n in 1:D
            val += grad_I[n] * mat[j, n]
        end
        return val * grad_J[i]
    end, Val(D * D)))
end

@inline function Internal_Product(a::AdjointGrad{D, DG},b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}) where 
    {D, DG, T_MAT, T_MEAS}
    data_a = a.g.data 
    data_b = b.data 
    return SMatrix{D, D, Float64}(ntuple(k -> begin
        i, j = (k - 1) % D + 1, (k - 1) ÷ D + 1
        grad_I = data_a[i].vec 
        grad_J = data_b[j].m.vec
        mat = data_b[j].mat
        dot_grad = 0.0
        @inbounds for n in 1:D
            dot_grad += grad_I[n] * grad_J[n]
        end
        return mat[j, i] * dot_grad
    end, Val(D * D)))
end


@inline function Internal_Product(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT1, T_MEAS1}}, 
                                  b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT2, T_MEAS2}}) where {D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2}
    data_a, data_b = a.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(linear_index -> begin
        k, m = (linear_index - 1) % D + 1, (linear_index - 1) ÷ D + 1 
        mat_a = data_a[k].mat
        mat_b = data_b[m].mat
        dot_grad = 0.0
        @inbounds for i in 1:D
            dot_grad += _get_val(data_a[k].m, i) * _get_val(data_b[m].m, i)
        end
        mat_dot = 0.0
        @inbounds for j in 1:D
            mat_dot += mat_a[k, j] * mat_b[m, j]
        end
        return dot_grad * mat_dot
    end, Val(D * D)))
end

@inline function Internal_Product(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT1, T_MEAS1}}, 
                                  b::VectorField{D, TransformedMeasure{D, DG, T_MAT2, T_MEAS2}}) where {D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2}
    data_a, data_b = a.data, b.data 
    return SMatrix{D, D, Float64}(ntuple(linear_index -> begin
        test_dof, base_dof = (linear_index - 1) % D + 1, (linear_index - 1) ÷ D + 1
        mat_a = data_a[test_dof].mat 
        mat_b = data_b[base_dof].mat      
        term1 = 0.0
        @inbounds for i in 1:D
            # Usamos _get_val para soportar DivergenceScalar o VecSingle indistintamente
            term1 += _get_val(data_a[test_dof].m, i) * mat_a[base_dof, i] 
        end
        term2 = 0.0
        @inbounds for j in 1:D
            term2 += mat_b[j, test_dof] * _get_val(data_b[base_dof].m, j)
        end     
        return term1 * term2
    end, Val(D * D)))
end

# Constructor para Integral
struct Integrand{F}
    f::F
end
const ∫ = Integrand

#Integrando un campo dado
@inline function Integrate(a::Integrand{<:AbstractVector}, b::Union{IntegrationSet, AbstractVector{<:IntegrationSet}})
    gs = b.gs
    jac = gs[:, end]
    weight = gs[:, end-1]
    return a.object .* (jac .* weight)
end

struct Integrated{T, M}
    arg::T
    b::M
end

#Local Valuation of Integrals for assembling matrices and vectors
@inline function Integrate(a::Integrand, b::M) where {M<:Union{IntegrationSet, AbstractVector{<:IntegrationSet}}}
    return Integrated(a.f, b)
end

struct MultiIntegrated
    terms::Vector{Integrated}
end

@inline (-)(a::Integrated) = Integrated((args...) -> -a.arg(args...), a.b)

@inline (+)(a::Integrated, b::Integrated) = MultiIntegrated([a, b])
@inline (+)(m::MultiIntegrated, a::Integrated) = MultiIntegrated(vcat(m.terms, [a]))
@inline (+)(a::Integrated, m::MultiIntegrated) = MultiIntegrated(vcat([a], m.terms))
@inline (+)(m1::MultiIntegrated, m2::MultiIntegrated) = MultiIntegrated(vcat(m1.terms, m2.terms))

@inline (-)(a::Integrated, b::Integrated) = a + (-b)
@inline (-)(m::MultiIntegrated, a::Integrated) = m + (-a)
@inline (-)(a::Integrated, m::MultiIntegrated) = a + (-m)
@inline (-)(m1::MultiIntegrated, m2::MultiIntegrated) = m1 + (-m2)


@inline function (*)(ag::AdjointGrad{D, T_MEAS}, f_uh::AnyEFGField) where {D, T_MEAS}
    return _apply_adjoint_transport(f_uh, ag.g)
end
@inline get_D(f::EFGFunction) = f.D
@inline get_D(f::VecEFGFunction) = f.source.D  # El wrapper mira a la madre
@inline get_D(f::TransposedVecEFGFunction) = f.D
@inline get_D(f::VolumetricEFGFunction) = f.D
@inline get_D(f::CompositeEFGFunction) = f.D
@inline get_D(f::ProductEFGFunction) = f.D
@inline get_D(f::ScalarFuncXEFGFunction) = f.D
@inline get_D(f::VectorFuncXEFGFunction) = f.D
@inline get_D(f::ScalarTimesId{T, D}) where {T, D} = D
@inline function (+)(a::AnyEFGField, b::AnyEFGField)
    D = get_D(a)
    L_guess = promote_type(eltype(a), eltype(b))
    L_new = L_guess === Any ? SMatrix{D, D, Float64, D*D} : L_guess 
    return CompositeEFGFunction{D, L_new, typeof(a), typeof(b)}(a, b, 1.0, 1.0, D)
end
@inline function (-)(a::AnyEFGField, b::AnyEFGField)
    D = get_D(a)
    L_guess = promote_type(eltype(a), eltype(b))
    L_new = L_guess === Any ? SMatrix{D, D, Float64, D*D} : L_guess 
    return CompositeEFGFunction{D, L_new, typeof(a), typeof(b)}(a, b, 1.0, -1.0, D)
end
@inline ()(a::VecSingleEFGMeasure{DG}, b::VecSingleEFGMeasure{DG}) where {DG} = a.vec * b.vec'
@inline ()(a::Integrand,b::Union{IntegrationSet, AbstractVector{<:IntegrationSet}}) = Integrate(a, b)

@inline (*)(a::Integrand,b::Union{IntegrationSet, AbstractVector{<:IntegrationSet}}) = Integrate(a, b)
@inline (*)(k::Real, s::SingleEFGMeasure{DG}) where DG = SingleEFGMeasure{DG}(k*s.phi, s.dphi, s.coord, s.ind)
@inline (*)(s::SingleEFGMeasure{DG}, k::Real) where DG = k * s
@inline (*)(k::Real, v::VecSingleEFGMeasure{DG}) where DG = VecSingleEFGMeasure{DG}(k*v.vec, v.coord, v.ind)
@inline (*)(v::VecSingleEFGMeasure{DG}, k::Real) where DG = k * v
@inline (*)(a::VecSingleEFGMeasure{DG}, b::VecSingleEFGMeasure{DG}) where DG = dot(a.vec, b.vec)
@inline (*)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
@inline (*)(a::SingleEFGMeasure{DG}, b::SingleEFGMeasure{DG}) where DG = (a.phi * b.phi)
@inline function (*)(a::EFGFunction{D, Float64, G}, b::SingleEFGMeasure{DG}) where {D, G, DG}
    val = evaluate_at_point(a, b)
    return SingleEFGMeasure{DG}(val * b.phi, b.dphi, b.coord, b.ind)
end
@inline (*)(b::SingleEFGMeasure, a::EFGFunction) = a * b
@inline function (*)(a::EFGFunction{D, Float64, G}, b::VecSingleEFGMeasure{DG}) where {D, G, DG}
    val = evaluate_at_point(a, b)
    return VecSingleEFGMeasure{DG}(val * b.vec, b.coord, b.ind)
end
@inline function (*)(a::VecEFGFunction, b::SingleEFGMeasure{DG}) where {DG}
    val = evaluate_at_point(a, b) 
    return VecSingleEFGMeasure{DG}(val * b.phi, b.coord, b.ind)
end
@inline function (*)(a::ScalarFuncXEFGFunction, b::SingleEFGMeasure{DG}) where {DG}
    val = evaluate_at_point(a, b)
    return SingleEFGMeasure{DG}(val * b.phi, b.dphi, b.coord, b.ind)
end
@inline function (*)(a::ScalarFuncXEFGFunction, b::VecSingleEFGMeasure{DG}) where {DG}
    val = evaluate_at_point(a, b)
    return VecSingleEFGMeasure{DG}(val * b.vec, b.coord, b.ind)
end
@inline function (*)(f::VolumetricEFGFunction{D, L, G, T_ORIG}, d::DivergenceScalar{D2, DG}) where {D, L, G, T_ORIG, D2, DG}
    return DivergenceScalar{D2, DG}(f * d.divfield)
end

@inline function (*)(d::DivergenceScalar{D2, DG}, f::VolumetricEFGFunction{D, L, G, T_ORIG}) where {D, L, G, T_ORIG, D2, DG}
    return f * d
end

@inline function (*)(f::VolumetricEFGFunction{D, L, G, T_ORIG}, v::VecSingleEFGMeasure{DG}) where {D, L, G, T_ORIG, DG}
    val_scalar = evaluate_at_point(f, v) 
    return VecSingleEFGMeasure{DG}(val_scalar * v.vec, v.coord, v.ind)
end
@inline (*)(v::VecSingleEFGMeasure{DG}, f::VolumetricEFGFunction{D, L, G, T_ORIG}) where {D, L, G, T_ORIG, DG} = f * v
@inline (*)(b::AnyEFGMeasure, a::ScalarFuncXEFGFunction) = a * b
@inline (*)(b::VecSingleEFGMeasure, a::EFGFunction) = a * b
@inline (*)(f::Function, b::SingleEFGMeasure{DG}) where DG = Internal_Product(f, b)
@inline (*)(b::SingleEFGMeasure{DG}, f::Function) where DG = Internal_Product(f, b)
@inline (*)(f::Function, b::VecSingleEFGMeasure{DG}) where DG = Internal_Product(f, b)
@inline (*)(b::VecSingleEFGMeasure{DG}, f::Function) where DG = Internal_Product(f, b)
@inline function (*)(comp::Composition, v::SingleEFGMeasure{DG}) where {DG}
    val = evaluate_at_point(comp, v)
    return SingleEFGMeasure{DG}(val * v.phi, v.dphi, v.coord, v.ind)
end
@inline (*)(v::SingleEFGMeasure{DG}, comp::Composition) where {DG} = comp * v
@inline function (*)(comp::Composition, v::VecSingleEFGMeasure{DG}) where {DG}
    val = evaluate_at_point(comp, v)
    return VecSingleEFGMeasure{DG}(val * v.vec, v.coord, v.ind)
end
@inline (*)(v::VecSingleEFGMeasure{DG}, comp::Composition) where {DG} = comp * v
@inline function (*)(comp::Composition, v::VectorField{D, SingleEFGMeasure{DG}}) where {D, DG}
    val = evaluate_at_point(comp, v.data[1])
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(i -> begin
        return SingleEFGMeasure{DG}(val * v.data[i].phi, v.data[i].dphi, v.data[i].coord, v.data[i].ind)
    end, Val(D)))
end
@inline (*)(v::VectorField{D, SingleEFGMeasure{DG}}, comp::Composition) where {D, DG} = comp * v
@inline (*)(a::DivergenceScalar{D,DG}, b::DivergenceScalar{D,DG}) where {D,DG} = Internal_Product(a, b)
@inline (*)(a::Real, v::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG} = VectorField(ntuple(i -> a * v[i], D)...)
@inline (*)(v::VectorField{D,SingleEFGMeasure{DG}}, a::Real) where {D,DG} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(a::VectorField{D,SingleEFGMeasure{DG}}, b::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG} = Internal_Product(a, b)
@inline (*)(a::Real, v::VectorField{D, VecSingleEFGMeasure{DG}}) where {D,DG} = 
    VectorField(ntuple(i -> a * v[i], Val(D))...)
@inline (*)(a::Real, ag::AdjointGrad{D,DG}) where {D,DG} = 
    AdjointGrad{D,DG}(a * ag.g)
@inline (*)(a::Real, d::DivergenceScalar{D,DG}) where {D,DG} = 
    DivergenceScalar{D,DG}(a * d.divfield)
@inline (*)(v::VectorField{D, VecSingleEFGMeasure{DG}}, a::Real) where {D,DG} = a * v
@inline (*)(ag::AdjointGrad, a::Real) = a * ag
@inline (*)(d::DivergenceScalar, a::Real) = a * d
@inline function (*)(c::Real, ls::LazySumGrads{D, DG}) where {D, DG}
    new_terms = map(t -> c * t, ls.terms)
    return LazySumGrads{D, DG, typeof(new_terms)}(new_terms)
end
@inline (*)(ls::LazySumGrads, c::Real) = c * ls
@inline (*)(v::VectorField{D,T}, a::Real) where {D,T<:Real} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(a::Real, v::VectorField{D,T}) where {D,T<:Real} = VectorField(ntuple(i -> v[i] * a, D)...)
@inline (*)(f::Function, α::Real) = x -> f(x) * α
@inline (*)(α::Real, f::Function) = x -> α * f(x)
@inline function (*)(c::Real, f::EFGFunction{D, L, G}) where {D, L, G}
    return EFGFunction{D, L, G}(f.field_nodal, f.PHI, f.DPHI, f.DOM, f.D, f.Measure, f.coeff * Float64(c),f.cache)
end
@inline (*)(f::EFGFunction, c::Real) = c * f

@inline function (*)(c::Real, f::VecEFGFunction{D, L, G, M, T_RES, F}) where {D, L, G, M, T_RES, F}
    return VecEFGFunction{D, L, G, M, T_RES, F}(f.source, f.coeff * Float64(c), f.cache)
end
@inline function (*)(c::Real, f::ScalarFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}) where {D, L, G, F, T, DG, L_OUT}
    return ScalarFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}(f.f, c * f.origin)
end
@inline (*)(f::ScalarFuncXEFGFunction, c::Real) = c * f
@inline (*)(f::VecEFGFunction, c::Real) = c * f

@inline function (*)(c::Real, f::VolumetricEFGFunction{D, L, G, T_ORIG}) where {D, L, G, T_ORIG}
    return VolumetricEFGFunction(c * f.origin, f.D)
end
@inline (*)(f::VolumetricEFGFunction, c::Real) = c * f

@inline function (*)(c::Real, f::TransposedVecEFGFunction{D, L, G, T_RES, F}) where {D, L, G, T_RES, F}
    return TransposedVecEFGFunction{D, L, G, T_RES, F}(c * f.origin, f.D)
end
@inline (*)(f::TransposedVecEFGFunction, c::Real) = c * f

@inline function (*)(c::Real, f::ProductEFGFunction{D, L, G, T1, T2}) where {D, L, G, T1, T2}
    return ProductEFGFunction{D, L, G, T1, T2}(f.f1, f.f2, f.D, f.coeff * Float64(c))
end

@inline (*)(f::ProductEFGFunction, c::Real) = c * f

@inline function (*)(c::Real, f::CompositeEFGFunction{D, L, T1, T2}) where {D, L, T1, T2}
    k = Float64(c)
    return CompositeEFGFunction{D, L, T1, T2}(f.f1, f.f2, f.c1 * k, f.c2 * k, f.D)
end
@inline (*)(f::CompositeEFGFunction, c::Real) = c * f

@inline function (*)(c::Real, s::ScalarTimesId{T, D}) where {T, D}
    return ScalarTimesId(c * s.f, D)
end

@inline (*)(s::ScalarTimesId, c::Real) = c * s

@inline function (*)(a::Real, v::VectorField{D, <:AdjointTransportMeasure}) where {D}
    return VectorField(ntuple(i -> a * v[i], Val(D))...)
end

@inline function (*)(a::Real, m::AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}) where {D, DG, T_MAT, T_MEAS}
    # Solo multiplicamos la matriz o la medida, en este caso la matriz es lo más directo
    return AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}(a * m.mat, m.m)
end

@inline (*)(v::VectorField{D, <:AdjointTransportMeasure}, a::Real) where {D} = a * v

@inline function (*)(s::ScalarTimesId{T, D}, v::VectorField{D, <:VecSingleEFGMeasure}) where {T, D}
    return s.f * v
end

import Base: eltype
@inline eltype(::EFGFunction{D, L, G}) where {D, L, G} = L
@inline eltype(::VecEFGFunction{D, L, G, M, T_RES, F}) where {D, L, G, M, T_RES, F} = T_RES
@inline eltype(::CompositeEFGFunction{D, L, T1, T2}) where {D, L, T1, T2} = L
@inline eltype(::TransposedVecEFGFunction{D, L, G, T_RES, F}) where {D, L, G, T_RES, F} = D == 1 ? Adjoint{Float64, SVector{G, Float64}} : SMatrix{G, D, Float64, G * D}
@inline eltype(::VolumetricEFGFunction{D, L, G, T_ORIG}) where {D, L, G, T_ORIG} = Float64
@inline eltype(::ScalarFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}) where {D, L, G, F, T, DG, L_OUT} = L_OUT
@inline eltype(::VectorFuncXEFGFunction{D, L, G, F, T, DG, L_OUT}) where {D, L, G, F, T, DG, L_OUT} = L_OUT
@inline eltype(::ScalarTimesId{T, D}) where {T, D} = SMatrix{D, D, Float64, D*D}

@inline (⋅)(f::Function, b::VecSingleEFGMeasure{DG}) where DG = Internal_Product(f, b)
@inline (⋅)(f::VecSingleEFGMeasure{DG}, b::Function) where DG = f⋅b
@inline (⋅)(a::EFGFunction, b::VecSingleEFGMeasure{DG}) where DG = Internal_Product(a, b)
@inline (⋅)(a::VecEFGFunction, b::SingleEFGMeasure{DG}) where DG = Internal_Product(a, b)
@inline (⋅)(a::EFGFunction, b::SingleEFGMeasure{DG}) where DG = Internal_Product(a, b)
@inline (⋅)(a::EFGFunction, b::EFGFunction) = Internal_Product(a, b)
@inline (⋅)(a::VecEFGFunction, b::VecEFGFunction) = Internal_Product(a, b)
@inline (⋅)(k::Real, v::VecSingleEFGMeasure{DG}) where DG = VecSingleEFGMeasure{DG}(k*v.vec, v.coord, v.ind)
@inline (⋅)(v::VecSingleEFGMeasure{DG}, k::Real) where DG = k * v
@inline function (⋅)(m::VecSingleEFGMeasure{DG}, f::VecEFGFunction) where DG
    val_f = evaluate_at_point(f, m) # Esto devuelve un SVector
    return SingleEFGMeasure{DG}(dot(val_f, m.vec),m.vec, m.coord, m.ind)
end
@inline (⋅)(f::VecEFGFunction, m::VecSingleEFGMeasure{DG}) where {DG} = m ⋅ f
@inline (⋅)(a::VecSingleEFGMeasure{DG}, b::VecSingleEFGMeasure{DG}) where DG = dot(a.vec, b.vec)
@inline (⋅)(v::VectorField{D,T}, g::VecSingleEFGMeasure{DG}) where {D,T,DG} =
    SingleEFGMeasure(sum(v[i] * g.vec[i] for i in 1:D), g.vec, g.coord,g.ind)
@inline (⋅)(a::VectorField{D,SingleEFGMeasure{DG}}, b::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG} = Internal_Product(a, b)
@inline function (*)(s::SingleEFGMeasure{DG}, v::VectorField{D, T}) where {DG, D, T}
    return VecSingleEFGMeasure{DG}(SVector{D, T}(v.data) * s.phi, s.coord, s.ind)
end
@inline function (*)(v::VectorField{D, T}, s::SingleEFGMeasure{DG}) where {DG, D, T}
    return s * v
end

@inline (⋅)(a::Int, b::Int) = a * b
@inline (⋅)(a::Int, b::Vector{Float64}) = a * b
@inline (⋅)(a::Vector{Float64}, b::Int) = a * b
@inline (⋅)(v::VectorField{D,SingleEFGMeasure{DG}}, f::VectorField{D,T}) where {D,T,DG} = VectorField(ntuple(i -> v[i] * f[i], D)...)
@inline (⋅)(f::VectorField{D,T}, v::VectorField{D,SingleEFGMeasure{DG}}) where {D,T<:Number,DG} = VectorField(ntuple(i -> v[i] * f[i], D)...)
@inline (⋅)(::Nabla, v::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG} = divergence(v)
@inline function (⋅)(f::Function, v::VectorField{D,SingleEFGMeasure{DG}}) where {D,DG}
    coord = v[1].coord
    val_f = f(coord)
    return VectorField(ntuple(i -> v[i] * val_f[i], D)...)
end
@inline (⋅)(v::VectorField{D,SingleEFGMeasure{DG}}, f::Function) where {D,DG} = f ⋅ v
@inline function (⋅)(comp::Composition, v::VectorField{D, SingleEFGMeasure{DG}}) where {D, DG}
    # Si la composición devuelve un vector (ej: una fuerza dependiente de u)
    val_vec = evaluate_at_point(comp, v.data[1])
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(i -> begin
        phi_val = val_vec[i] * v.data[i].phi
        return SingleEFGMeasure{DG}(phi_val, v.data[i].dphi, v.data[i].coord, v.data[i].ind)
    end, Val(D)))
end
@inline (⋅)(v::VectorField{D, SingleEFGMeasure{DG}}, comp::Composition) where {D, DG} = comp ⋅ v

@inline (⊙)(a::VectorField{D,VecSingleEFGMeasure{DG}}, b::VectorField{D,VecSingleEFGMeasure{DG}}) where {D,DG} = Internal_Product(a, b)
@inline (⊙)(a::VectorField{D,VecSingleEFGMeasure{DG}}, b::AdjointGrad{D,DG}) where {D,DG} = Internal_Product(a, b)
@inline (⊙)(a::AdjointGrad{D,DG},b::VectorField{D,VecSingleEFGMeasure{DG}}) where {D,DG} = transpose(Internal_Product(b, a))
@inline (⊙)(a::AdjointGrad{D,DG}, b::AdjointGrad{D,DG}) where {D,DG} = Internal_Product(a,b)
@inline function (⋅)(f::AnyEFGField, v::VectorField{D, SingleEFGMeasure{DG}}) where {D, DG}
    val_uh = evaluate_at_point(f, v.data[1]) 
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(i -> begin
        phi_val = val_uh[i] * v.data[i].phi
        return SingleEFGMeasure{DG}(
            phi_val, 
            v.data[i].dphi,
            v.data[1].coord, 
            v.data[1].ind
        )
    end, Val(D)))
end
@inline (⋅)(v::VectorField{D, SingleEFGMeasure{DG}}, f::AnyEFGField) where {D, DG} = f ⋅ v
@inline function (⊙)(f::AnyEFGField, v::VectorField{D, VecSingleEFGMeasure{DG}}) where {D, DG}
    mat_uh = evaluate_at_point(f, v.data[1])
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(i -> begin
        phi_val = sum(ntuple(j -> mat_uh[i, j] * v.data[j].vec[j], Val(D)))
        return SingleEFGMeasure{DG}(phi_val,v.data[i].vec,v.data[1].coord, v.data[1].ind)
    end, Val(D)))
end
@inline (⊙)(v::VectorField{D, VecSingleEFGMeasure{DG}}, f::AnyEFGField) where {D, DG} = f⊙v
@inline function (⊙)(adj::AdjointGrad{D,DG}, f::AnyEFGField) where {D, DG}
    mat_uh = evaluate_at_point(f, adj.g.data[1])
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(j -> begin
        phi_val = sum(ntuple(i -> adj.g.data[j].vec[i] * mat_uh[i, j], Val(DG)))
        return SingleEFGMeasure{DG}(phi_val,adj.g.data[j].vec,adj.g.data[1].coord,adj.g.data[1].ind)
    end, Val(D)))
end
@inline (⊙)(f::AnyEFGField, adj::AdjointGrad) = adj ⊙ f
@inline function (⊙)(f::AnyEFGField, v::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}) where {D, DG, T_MAT, T_MEAS}
    mat_B = evaluate_at_point(f, v.data[1].m)
    mat_A = v.data[1].mat                     
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(k -> begin
        grad_phi = v.data[k].m.vec
        phi_val = 0.0
        @inbounds for i in 1:D, j in 1:D
            phi_val += mat_A[k, j] * grad_phi[i] * mat_B[i, j]
        end
        return SingleEFGMeasure{DG}(phi_val, grad_phi, v.data[1].m.coord, v.data[1].m.ind)
    end, Val(D)))
end
@inline (⊙)(v::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}, f::AnyEFGField) where {D, DG, T_MAT, T_MEAS} = f ⊙ v
@inline function (⊙)(f::AnyEFGField, v::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}) where {D, DG, T_MAT, T_MEAS}
    mat_B = evaluate_at_point(f, v.data[1].m)
    mat_A = v.data[1].mat
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(k -> begin
        grad_phi = v.data[k].m.vec
        phi_val = 0.0
        @inbounds for i in 1:D, j in 1:D
            phi_val += mat_A[i, k] * mat_B[i, j] * grad_phi[j]
        end
        return SingleEFGMeasure{DG}(phi_val, grad_phi, v.data[1].m.coord, v.data[1].m.ind)
    end, Val(D)))
end
@inline (⊙)(v::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}, f::AnyEFGField) where {D, DG, T_MAT, T_MEAS} = f ⊙ v
@inline (⊙)(a::VectorField{D, VecSingleEFGMeasure{DG}}, b::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}) where
{D, DG, T_MAT, T_MEAS} = Internal_Product(a, b)
@inline (⊙)(a::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}, b::VectorField{D, VecSingleEFGMeasure{DG}}) where
{D, DG, T_MAT, T_MEAS} = transpose(Internal_Product(b, a))
@inline (⊙)(a::VectorField{D, VecSingleEFGMeasure{DG}}, b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}) where 
{D, DG, T_MAT, T_MEAS} = Internal_Product(a, b)
@inline (⊙)(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}, b::VectorField{D, VecSingleEFGMeasure{DG}}) where
{D, DG, T_MAT, T_MEAS} = transpose(Internal_Product(b, a))
@inline (⊙)(a::AdjointGrad{D, DG}, b::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}) where 
{D, DG, T_MAT, T_MEAS} = Internal_Product(a,b)
@inline (⊙)(a::VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}} ,b::AdjointGrad{D, DG}) where 
{D, DG, T_MAT, T_MEAS} = transpose(Internal_Product(b,a))
@inline (⊙)(a::AdjointGrad{D, DG},b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}) where 
{D, DG, T_MAT, T_MEAS} = Internal_Product(a,b)
@inline (⊙)(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT, T_MEAS}}, b::AdjointGrad{D, DG}) where 
{D, DG, T_MAT, T_MEAS} = transpose(Internal_Product(b,a))
@inline (⊙)(a::AnyGradTerm{D, DG}, b::TransformedMeasure{D, DG, T_MAT, <:DivergenceScalar}) where 
{D, DG, T_MAT} = Internal_Product(a,b)
@inline (⊙)(a::TransformedMeasure{D, DG, T_MAT, <:DivergenceScalar}, b::AnyGradTerm{D, DG}) where
{D, DG, T_MAT} = transpose(Internal_Product(b,a))
@inline (⊙)(a::VectorField{D, TransformedMeasure{D, DG, T_MAT1, T_MEAS1}}, b::VectorField{D, TransformedMeasure{D, DG, T_MAT2, T_MEAS2}}) where 
{D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2} = Internal_Product(a,b)
@inline (⊙)(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT1, T_MEAS1}}, b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT2, T_MEAS2}}) where 
{D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2} = Internal_Product(a,b)
@inline (⊙)(a::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT1, T_MEAS1}}, b::VectorField{D, TransformedMeasure{D, DG, T_MAT2, T_MEAS2}}) where 
{D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2} = Internal_Product(a,b)
@inline (⊙)(a::VectorField{D, TransformedMeasure{D, DG, T_MAT2, T_MEAS2}}, b::VectorField{D, AdjointTransportMeasure{D, DG, T_MAT1, T_MEAS1}}) where 
{D, DG, T_MAT1, T_MEAS1, T_MAT2, T_MEAS2} = transpose(Internal_Product(b,a))
@inline function (*)(c::Real, v::VectorField{D, TransformedMeasure{D, DG, T_MAT,T_MEAS}}) where {D, DG, T_MAT,T_MEAS}
    return VectorField{D, TransformedMeasure{D, DG, T_MAT, T_MEAS}}(
        ntuple(i -> TransformedMeasure{D, DG, T_MAT,T_MEAS}(c * v.data[i].mat, v.data[i].m), Val(D)))
end

@inline (*)(v::VectorField{D, TransformedMeasure{D, DG, T_MAT,T_MEAS}}, c::Real) where {D, DG, T_MAT,T_MEAS} = c * v
@inline (*)(a::DivergenceScalar, b::VectorField{D, <:TransformedMeasure}) where {D} = Internal_Product(a, b)
@inline (*)(b::VectorField{D, <:TransformedMeasure}, a::DivergenceScalar) where {D} = Internal_Product(a, b)


@inline function (⊙)(a::LazySumGrads{D,DG}, b::LazySumGrads{D,DG}) where {D,DG}
    return sum(map(ta -> sum(map(tb -> ta ⊙ tb, b.terms)), a.terms))
end

@inline function (⊙)(a::AnyGradTerm{D,DG}, b::LazySumGrads{D,DG}) where {D,DG}
    return sum(map(tb -> a ⊙ tb, b.terms))
end

@inline function (⊙)(a::LazySumGrads{D,DG}, b::AnyGradTerm{D,DG}) where {D,DG}
    return sum(map(ta -> ta ⊙ b, a.terms))
end

@inline _lsg_contract_distribute(terms::Tuple{T}, f::AnyEFGField) where {T} = terms[1] ⊙ f
@inline function _lsg_contract_distribute(terms::T_TUPLE, f::AnyEFGField) where {T_TUPLE<:Tuple}
    return terms[1] ⊙ f + _lsg_contract_distribute(Base.tail(terms), f)
end

@inline (⊙)(lsg::LazySumGrads, f::AnyEFGField) = _lsg_contract_distribute(lsg.terms, f)
@inline (⊙)(f::AnyEFGField, lsg::LazySumGrads) = _lsg_contract_distribute(lsg.terms, f)

@inline (*)(ls::LazySumGrads{D,DG}, id::IdentityOperator) where {D,DG} = 
    let new_terms = map(t -> t * id, ls.terms)
        LazySumGrads{D, DG, typeof(new_terms)}(new_terms)
    end

@inline (*)(ls::LazySumGrads{D,DG}, s::ScalarTimesId) where {D,DG} = 
    let new_terms = map(t -> t * s, ls.terms)
        LazySumGrads{D, DG, typeof(new_terms)}(new_terms)
    end

# 1. Estructura genérica que envuelve al operador Nabla
@inline (+)(m::SMatrix{D,D,T}, s::Number) where {D,T} = m + s * I
@inline (+)(s::Number, m::SMatrix{D,D,T}) where {D,T} = m + s * I
@inline function (+)(a::SingleEFGMeasure{DG}, b::SingleEFGMeasure{DG}) where DG
    return SingleEFGMeasure{DG}(a.phi + b.phi, a.dphi+ b.dphi, a.coord, a.ind)
end
@inline function (+)(v1::VectorField{D, SingleEFGMeasure{DG}}, 
                           v2::VectorField{D, SingleEFGMeasure{DG}}) where {D, DG}
    return VectorField{D, SingleEFGMeasure{DG}}(ntuple(i -> v1.data[i] + v2.data[i], Val(D)))
end
# Nothing Operations

@inline (+)(::Nothing, ::Nothing) = nothing
@inline (-)(a::Nothing) = nothing
@inline (-)(a::Nothing, b::Nothing) = nothing
@inline (*)(::Nothing, ::Nothing) = nothing
@inline (⋅)(::Nothing, ::Nothing) = nothing
@inline (*)(::Any, ::Nothing) = nothing
@inline (*)(::Nothing,::Any) = nothing
@inline (⋅)(::Any, ::Nothing) = nothing
@inline (⋅)(::Nothing,::Any) = nothing
@inline (⋅)(::Nothing, ::VectorField{D,T}) where {D,T<:Number} = nothing
@inline (⋅)(::VectorField{D,T}, ::Nothing,) where {D,T<:Number} = nothing
@inline adjoint(::Nothing) = nothing
@inline ⊙(::Nothing, ::Nothing) = nothing
@inline ⊙(::Nothing, ::Float64) = nothing
@inline ⊙(::Float64, ::Nothing) = nothing
@inline (⋅)(::Nothing, ::Function) = nothing
@inline (⋅)(::Function, ::Nothing) = nothing
@inline (*)(::Nothing, ::Function) = nothing
@inline (*)(::Function, ::Nothing) = nothing
@inline (*)(::Nothing, ::EFGFunction) = nothing
@inline (*)(::EFGFunction, ::Nothing) = nothing
@inline (⊙)(f::VecEFGFunction, ::Nothing) = nothing
@inline (⊙)(::Nothing, v::VectorField) = nothing
@inline (*)(::Nothing, ::DivergenceScalar) = nothing
@inline (*)(::DivergenceScalar, ::Nothing) = nothing
@inline tr(::Nothing) = nothing
@inline (*)(a::Number, ::IdentityOperator) = a
@inline (*)(::IdentityOperator, a::Number) = a