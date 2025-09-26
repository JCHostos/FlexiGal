function pgauss(k::Int)
    if k == 1
        # 1 punto de Gauss: fila 1 = puntos, fila 2 = pesos
        v = [0.0;
            2.0]
    elseif k == 2
        v = [-1/sqrt(3) 1/sqrt(3);
            1.0 1.0]
    elseif k == 3
        x1 = -sqrt(0.6)
        w1 = 5 / 9
        v = [x1 0.0 -x1;
            w1 8/9 w1]
    elseif k == 4
        x1 = -0.8611363115940526
        x2 = -0.3399810435848563
        w1 = 0.3478548451374539
        w2 = 0.6521451548625461
        v = [x1 x2 -x2 -x1;
            w1 w2 w2 w1]
    else
        error("pgauss solo está implementado para k = 1, 2, 3 o 4")
    end
    return v
end

function Measure2D_Cell(xi::Float64, eta::Float64, xe::Array{Float64,3})
    numcell, numvxcell, dim = size(xe)
    @assert numvxcell == 4 "Elemento cuadrilátero de 4 nodos esperado"

    # ----- Funciones de forma -----
    phi = [
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ]

    dphidxi = [
        -0.25 * (1 - eta),
        0.25 * (1 - eta),
        0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ]

    dphideta = [
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
        0.25 * (1 + xi),
        0.25 * (1 - xi)
    ]

    # ----- Coordenadas físicas y derivadas -----
    xg = zeros(numcell, dim)
    dxgxi = zeros(numcell, dim)
    dxgeta = zeros(numcell, dim)

    for i in 1:dim          # coordenada física (x,y[,z])
        for k in 1:numvxcell
            xg[:, i] .+= phi[k] .* xe[:, k, i]
            dxgxi[:, i] .+= dphidxi[k] .* xe[:, k, i]
            dxgeta[:, i] .+= dphideta[k] .* xe[:, k, i]
        end
    end

    # ----- Jacobiano -----
    Jac = if dim == 2
        # Elemento plano: determinante de la matriz 2x2
        dxgxi[:, 1] .* dxgeta[:, 2] .- dxgxi[:, 2] .* dxgeta[:, 1]
    elseif dim == 3
        # Superficie en 3D: norma del producto vectorial
        Sx = dxgxi[:, 2] .* dxgeta[:, 3] .- dxgeta[:, 2] .* dxgxi[:, 3]
        Sy = dxgeta[:, 1] .* dxgxi[:, 3] .- dxgxi[:, 1] .* dxgeta[:, 3]
        Sz = dxgxi[:, 1] .* dxgeta[:, 2] .- dxgeta[:, 1] .* dxgxi[:, 2]
        sqrt.(Sx .^ 2 .+ Sy .^ 2 .+ Sz .^ 2)
    else
        error("dim debe ser 2 o 3")
    end

    return xg, Jac, phi
end

function Measure3D_Cell(xi::Float64, eta::Float64, zeta::Float64, xe::Array{Float64,3})
    # XYE: numcell × numvxcell × dim  (dim = 3)
    numcell, numvxcell, dim = size(xe)
    @assert dim == 3 "Esta función es sólo para celdas 3D"

    # --- Funciones de forma hexaédricas (8 nodos) ---
    phi = [
        (1 / 8) * (1 - xi) * (1 - eta) * (1 - zeta),
        (1 / 8) * (1 + xi) * (1 - eta) * (1 - zeta),
        (1 / 8) * (1 + xi) * (1 + eta) * (1 - zeta),
        (1 / 8) * (1 - xi) * (1 + eta) * (1 - zeta),
        (1 / 8) * (1 - xi) * (1 - eta) * (1 + zeta),
        (1 / 8) * (1 + xi) * (1 - eta) * (1 + zeta),
        (1 / 8) * (1 + xi) * (1 + eta) * (1 + zeta),
        (1 / 8) * (1 - xi) * (1 + eta) * (1 + zeta)
    ]

    dphidxi = [
        -(1 / 8) * (1 - eta) * (1 - zeta),
        (1 / 8) * (1 - eta) * (1 - zeta),
        (1 / 8) * (1 + eta) * (1 - zeta),
        -(1 / 8) * (1 + eta) * (1 - zeta),
        -(1 / 8) * (1 - eta) * (1 + zeta),
        (1 / 8) * (1 - eta) * (1 + zeta),
        (1 / 8) * (1 + eta) * (1 + zeta),
        -(1 / 8) * (1 + eta) * (1 + zeta)
    ]

    dphideta = [
        -(1 / 8) * (1 - xi) * (1 - zeta),
        -(1 / 8) * (1 + xi) * (1 - zeta),
        (1 / 8) * (1 + xi) * (1 - zeta),
        (1 / 8) * (1 - xi) * (1 - zeta),
        -(1 / 8) * (1 - xi) * (1 + zeta),
        -(1 / 8) * (1 + xi) * (1 + zeta),
        (1 / 8) * (1 + xi) * (1 + zeta),
        (1 / 8) * (1 - xi) * (1 + zeta)
    ]

    dphidzeta = [
        -(1 / 8) * (1 - xi) * (1 - eta),
        -(1 / 8) * (1 + xi) * (1 - eta),
        -(1 / 8) * (1 + xi) * (1 + eta),
        -(1 / 8) * (1 - xi) * (1 + eta),
        (1 / 8) * (1 - xi) * (1 - eta),
        (1 / 8) * (1 + xi) * (1 - eta),
        (1 / 8) * (1 + xi) * (1 + eta),
        (1 / 8) * (1 - xi) * (1 + eta)
    ]

    # Gradientes en coordenadas locales (3 × 8)
    dphiloc = vcat(dphidxi', dphideta', dphidzeta')

    # --- Inicialización ---
    J = zeros(numcell, dim, dim)   # Jacobiano de cada celda
    xg = zeros(numcell, dim)        # Coordenadas físicas en el punto (xi,eta,zeta)

    # Ensamblaje de J y cálculo de XYG
    for i in 1:dim          # dirección física (x,y,z)
        for j in 1:dim      # derivada respecto a (xi,eta,zeta)
            for k in 1:numvxcell
                J[:, i, j] .+= dphiloc[j, k] .* xe[:, k, i]
                if j == 1   # sólo una vez por nodo para XYG
                    xg[:, i] .+= phi[k] .* xe[:, k, i]
                end
            end
        end
    end

    # --- Determinante del Jacobiano (volumen local) ---
    Jac = @. J[:,1,1]*(J[:,2,2]*J[:,3,3] - J[:,3,2]*J[:,2,3]) -
        J[:,1,2]*(J[:,2,1]*J[:,3,3] - J[:,3,1]*J[:,2,3]) +
        J[:,1,3]*(J[:,2,1]*J[:,3,2] - J[:,3,1]*J[:,2,2])

    return xg, Jac
end

function egauss(xc::Array{Float64,2}, conn::Array{Int,2}, gauss::Array{Float64,2})
    dim = size(xc, 2)                  # 2D o 3D
    l = size(gauss, 2)                # número de puntos de Gauss por dirección
    npts = l^dim                       # total de puntos de Gauss por celda
    numcell = size(conn, 1)            # número de celdas
    # columnas: x,y,(z), peso, jac
    ncol = dim + 2                     # 2D -> 4, 3D -> 5
    gs = zeros(numcell * npts, ncol)

    xe = Coords_by_Ele(xc, conn)             # nodos por celda

    count = 0
    # generamos todos los índices combinados de Gauss
    if dim == 2
        for j in 1:l
            for i in 1:l
                count += 1
                xi, eta = gauss[1, i], gauss[1, j]
                xg, Jac = Measure2D_Cell(xi, eta, xe)
                gs[count:l^2:end, 1:2] = xg[:, 1:2]
                gs[count:l^2:end, 3] .= gauss[2, i] * gauss[2, j]
                gs[count:l^2:end, 4] .= Jac
            end
        end
    elseif dim == 3
        for k in 1:l, j in 1:l, i in 1:l
            count += 1
            xi, eta, zeta = gauss[1, i], gauss[1, j], gauss[1, k]
            xg, Jac = Measure3D_Cell(xi, eta, zeta, xe)
            gs[count:l^3:end, 1:3] = xg
            gs[count:l^3:end, 4] .= gauss[2, i] * gauss[2, j] * gauss[2, k]
            gs[count:l^3:end, 5] .= Jac
        end
    else
        error("Solo 2D o 3D soportados")
    end

    return gs
end

function egauss_bound(xc::Array{Float64,2}, conn::Array{Int,2}, gauss::Array{Float64,2})
    dim = size(xc, 2)                  # 2D o 3D
    l = size(gauss, 2)                # número de puntos de Gauss por dirección
    npts = l^(dim-1)                       # total de puntos de Gauss por celda
    numcell = size(conn, 1)            # número de celdas
    # columnas: x,y,(z), peso, jac
    ncol = dim + 2                     # 2D -> 4, 3D -> 5
    gs = zeros(numcell * npts, ncol)
    xe = Coords_by_Ele(xc, conn)             # nodos por celda
    count = 0
    # generamos todos los índices combinados de Gauss
    if dim == 2
            for i in 1:l
                count += 1
                xi = gauss[1, i]
                xg = zeros(numcell, 2)  # n elementos x 2 coordenadas
                xg[:, 1] .= (1 - xi)/2 * xe[:, 1, 1] .+ (1 + xi)/2 * xe[:, 2, 1]
                xg[:, 2] .= (1 - xi)/2 * xe[:, 1, 2] .+ (1 + xi)/2 * xe[:, 2, 2]
                gs[count:l:end,1:2] .= xg[:,1:2]
                gs[count:l:end, 3] .= gauss[2, i]
                gs[count:l:end, 4] .= sqrt.((xe[:,1,1]-xe[:,2,1]) .^ 2 + (xe[:,2,2]-xe[:,1,2]) .^ 2) / 2;
            end
    elseif dim == 3
        for j in 1:l, i in 1:l
            count += 1
            xi, eta = gauss[1, i], gauss[1, j]
            xg, Jac = Measure2D_Cell(xi, eta, xe)
            gs[count:l^2:end, 1:3] .= xg
            gs[count:l^2:end, 4] .= gauss[2, i] * gauss[2, j]
            gs[count:l^2:end, 5] .= Jac
        end
    else
        error("Solo 2D o 3D soportados")
    end

    return gs
end