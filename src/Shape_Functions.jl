using LinearAlgebra
"""
    domain(gpos::AbstractVector, x::AbstractMatrix, dm::AbstractMatrix) -> Vector{Int}

Devuelve los índices de los nodos cuyo dominio de influencia encierra a `gpos`.

- `gpos` : coordenadas del punto gauss (longitud d, d=2 o 3)
- `x`    : matriz nnodes × d con las coordenadas de los nodos
- `dm`   : matriz nnodes × d con los semianchos por nodo y dirección
"""
function domain(gpos::AbstractVector{T}, x::AbstractMatrix{T}, dm::AbstractMatrix{T}) where T<:Real
    nnodes, dim = size(x)
    inside = trues(nnodes)  # vector booleano pre-alocado

    @inbounds for j in 1:dim
        colx = @view x[:, j]
        coldm = @view dm[:, j]
        g = gpos[j]
        for i in 1:nnodes
            # marcamos falso si la diferencia absoluta excede el delta
            inside[i] &= abs(colx[i] - g) <= coldm[i]
        end
    end

    return findall(inside)
end
"""
    cubwgt(dif, v, dm)

Calcula los **pesos cúbicos tipo spline** y sus derivadas parciales para los nodos
relevantes de un punto de evaluación (por ejemplo, un punto de Gauss).

# Argumentos
- `dif::Array{Float64,2}`: Diferencias entre el punto de evaluación y las coordenadas
  de los nodos seleccionados, tamaño `length(v) × dim` (dim = 2 o 3).
- `v::Vector{Int}`: Índices de los nodos cuya influencia se está evaluando.
- `dm::Array{Float64,2}`: Tamaños de dominio de influencia de cada nodo, con dimensiones
  `n_nodes × dim`.

# Salidas
- `w::Vector{Float64}`: Vector de pesos cúbicos evaluados en cada nodo.
- `dw::Array{Float64,2}`: Matriz de derivadas parciales de los pesos, tamaño
  `length(v) × dim`.
# Descripción
Para cada nodo `i`:
1. Se normaliza la distancia por la dimensión del dominio de influencia: `r = |dif|/dm`.
2. Se evalúa la función spline cúbica:
   - Para `r <= 0.5` se usa la forma suave interior.
   - Para `r > 0.5` se usa la forma exterior.
3. Se calcula el producto de los pesos por dirección para obtener el peso final
   `w(i)`.
4. Se calculan las derivadas parciales usando la regla del producto.

Esta función permite generalizar los cálculos de pesos y derivadas de forma
consistente tanto para 2D como para 3D.
"""
function cubwgt(dif, v, dm)
    l, dim = size(dif)  # dim = 2 o 3
    w = zeros(l)
    dw = zeros(l, dim)

    for i in 1:l
        wtimes = zeros(dim)
        dwtimes = zeros(dim)
        for d in 1:dim
            dr = sign(dif[i, d]) / dm[v[i], d]
            r = abs(dif[i, d]) / dm[v[i], d]
            if r > 0.5
                wtimes[d] = (4 / 3) - 4 * r + 4 * r^2 - (4 / 3) * r^3
                dwtimes[d] = (-4 + 8 * r - 4 * r^2) * dr
            else
                wtimes[d] = (2 / 3) - 4 * r^2 + 4 * r^3
                dwtimes[d] = (-8 * r + 12 * r^2) * dr
            end
        end
        if dim == 2
            w[i] = wtimes[1] * wtimes[2]
            dw[i, 1] = dwtimes[1] * wtimes[2]
            dw[i, 2] = wtimes[1] * dwtimes[2]
        else  # dim == 3
            w[i] = wtimes[1] * wtimes[2] * wtimes[3]
            dw[i, 1] = dwtimes[1] * wtimes[2] * wtimes[3]
            dw[i, 2] = wtimes[1] * dwtimes[2] * wtimes[3]
            dw[i, 3] = wtimes[1] * wtimes[2] * dwtimes[3]
        end
    end
    return w, dw
end
"""
    shape_general(gpos, x, v, dm)

Calcula las funciones de forma y sus derivadas para nodos de influencia `v` en 2D o 3D.

# Argumentos
- `gpos` : vector de posición del punto de Gauss (2D o 3D)
- `x`    : coordenadas de los nodos (n_nodos x dim)
- `v`    : índices de nodos de influencia
- `dm`   : dominios de influencia de los nodos (n_nodos x dim)

# Retorna
- `phi`    : vector de funciones de forma en `gpos`
- `dphi`   : matriz de derivadas de `phi` con respecto a cada coordenada (dim x n_nodos)
"""
function Improved_Moving_Least_Squares(gpos, x, v, dm)
    L = length(v)
    dim = length(gpos)        # 2 o 3
    nv = x[v, :]              # nodos de influencia

    # Pre-alocar q: filas = L, columnas según polinomio
    if dim == 2
        q = Array{Float64}(undef, L, 4)  # 1, x, y, xy
        @inbounds for i in 1:L
            q[i, 1] = 1.0
            q[i, 2] = nv[i, 1]
            q[i, 3] = nv[i, 2]
            q[i, 4] = nv[i, 1] * nv[i, 2]
        end
    elseif dim == 3
        q = Array{Float64}(undef, L, 8)  # 1, x, y, z, xy, xz, yz, xyz
        @inbounds for i in 1:L
            x_, y_, z_ = nv[i, 1], nv[i, 2], nv[i, 3]
            q[i, 1] = 1.0
            q[i, 2] = x_
            q[i, 3] = y_
            q[i, 4] = z_
            q[i, 5] = x_ * y_
            q[i, 6] = x_ * z_
            q[i, 7] = y_ * z_
            q[i, 8] = x_ * y_ * z_
        end
    else
        error("Solo 2D o 3D soportados")
    end

    # Pre-alocar dif (L × dim)
    dif = Array{Float64}(undef, L, dim)
    @inbounds for i in 1:L
        for d in 1:dim
            dif[i, d] = gpos[d] - nv[i, d]
        end
    end
    w, dw = cubwgt(dif, v, dm)  # w: L, dw: L x dim

    # Polinomio en gpos
    if dim == 2
        QQg = [1 gpos' gpos[1]*gpos[2];
            0 1 0 gpos[2];
            0 0 1 gpos[1]]
    else
        QQg = [1 gpos' gpos[1]*gpos[2] gpos[1]*gpos[3] gpos[2]*gpos[3] gpos[1]*gpos[2]*gpos[3];
            0 1 0 0 gpos[2] gpos[3] 0 gpos[2]*gpos[3];
            0 0 1 0 gpos[1] 0 gpos[3] gpos[1]*gpos[3];
            0 0 0 1 0 gpos[1] gpos[2] gpos[1]*gpos[2]]
    end

    # Gram-Schmidt ortogonalizado
    npoly = size(q, 2)
    p = copy(q)
    PPg = copy(QQg[:, 1:npoly])
    for i in 2:npoly
        for k in 1:i-1
            c1 = (q[:, i]' * (w .* p[:, k])) / (p[:, k]' * (w .* p[:, k]))
            p[:, i] -= c1 * p[:, k]
            PPg[:, i] -= c1 * PPg[:, k]
        end
    end

    aa = p' * (w .* p)
    aia = (1.0 ./ diag(aa))

    # Calcular gam
    gam = zeros(npoly, dim + 1)
    gam[:, 1] = aia .* PPg[1, :]
    for d in 1:dim
        daa = p' * (dw[:,d] .* p)
        gam[:, d+1] = aia .* (PPg[d+1, :] - daa * gam[:, 1])
    end
    # Construir phi y derivadas
    B = p .* w
    phi = (B * gam[:, 1])
    dphi = zeros(L,dim)
    for d in 1:dim
        dphi[:,d] = ((B * gam[:, d+1]) + ((p .* dw[:, d]) * gam[:, 1]))
    end

    return phi, dphi
end
"""
    MLS_SHAPE_FUN(gs, xy, dm, numqc)

Calcula las funciones de forma tipo MLS para cada punto de Gauss.

# Argumentos
- `gs`    : matriz de puntos de Gauss (numqc × dim)
- `xy`    : coordenadas de los nodos (nnodes × dim)
- `dm`    : dominios de influencia de los nodos (nnodes × dim)
- `numqc` : número de puntos de Gauss

# Retorna
- `PHI`   : Vector de vectores de funciones de forma por punto de Gauss
- `DPHI`  : Vector de vectores de derivadas de PHI por punto de Gauss (cada elemento: dim × nvec)
- `DOM`   : Vector de vectores con índices de nodos de influencia por punto de Gauss
"""
function SHAPE_FUN(gs, x, dm)
    # Inicializar vectores de resultados
    numqc = size(gs, 1)
    dim = size(x, 2)  # 2 o 3
    PHI = [Float64[] for _ in 1:numqc]
    DPHI = [zeros(Float64, 0, 0) for _ in 1:numqc]
    DOM = [Int[] for _ in 1:numqc]

    @inbounds for (ind, gg) in enumerate(eachrow(gs))
        gpos = gg[1:dim]
        v = domain(gpos, x, dm)   # nodos de influencia para este punto

        # Función de forma y derivadas (shape_general devuelve phi y dphi)
        phi, dphi = Improved_Moving_Least_Squares(gpos, x, v, dm)

        # Guardar resultados
        PHI[ind] = phi
        DPHI[ind] = dphi
        DOM[ind] = v
    end

    return PHI, DPHI, DOM
end
