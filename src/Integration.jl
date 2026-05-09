function pgauss(k::Int)
    if k == 1
        v = reshape([0.0, 2.0], 2, 1)
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
        error("Gauss integrations just implemented for ngpts = 1, 2, 3 o 4")
    end
    return v
end

function Measure2D_Cell(xi::Float64, eta::Float64, xe::Array{Float64,3})
    numcell, numvxcell, dim = size(xe)
    @assert numvxcell == 4 "Elemento cuadrilátero de 4 nodos esperado"

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

    xg = zeros(numcell, dim)
    dxgxi = zeros(numcell, dim)
    dxgeta = zeros(numcell, dim)

    for i in 1:dim 
        for k in 1:numvxcell
            xg[:, i] .+= phi[k] .* xe[:, k, i]
            dxgxi[:, i] .+= dphidxi[k] .* xe[:, k, i]
            dxgeta[:, i] .+= dphideta[k] .* xe[:, k, i]
        end
    end

    Jac = if dim == 2
        dxgxi[:, 1] .* dxgeta[:, 2] .- dxgxi[:, 2] .* dxgeta[:, 1]
    elseif dim == 3
        Sx = dxgxi[:, 2] .* dxgeta[:, 3] .- dxgeta[:, 2] .* dxgxi[:, 3]
        Sy = dxgeta[:, 1] .* dxgxi[:, 3] .- dxgxi[:, 1] .* dxgeta[:, 3]
        Sz = dxgxi[:, 1] .* dxgeta[:, 2] .- dxgeta[:, 1] .* dxgxi[:, 2]
        sqrt.(Sx .^ 2 .+ Sy .^ 2 .+ Sz .^ 2)
    else
        error("dim must be 2 o 3")
    end

    return xg, Jac, phi
end

function Measure3D_Cell(xi::Float64, eta::Float64, zeta::Float64, xe::Array{Float64,3})
    numcell, numvxcell, dim = size(xe)
    @assert dim == 3 "Esta función es sólo para celdas 3D"

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

    dphiloc = vcat(dphidxi', dphideta', dphidzeta')

    J = zeros(numcell, dim, dim)
    xg = zeros(numcell, dim)

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

    Jac = @. J[:,1,1]*(J[:,2,2]*J[:,3,3] - J[:,3,2]*J[:,2,3]) -
        J[:,1,2]*(J[:,2,1]*J[:,3,3] - J[:,3,1]*J[:,2,3]) +
        J[:,1,3]*(J[:,2,1]*J[:,3,2] - J[:,3,1]*J[:,2,2])

    return xg, Jac
end

function egauss(xc::Array{Float64,2}, conn::Array{Int,2}, gauss::AbstractMatrix{Float64})
    dim = size(xc, 2)                 
    l = size(gauss, 2)                
    npts = l^dim                     
    numcell = size(conn, 1)  
    ncol = dim + 2          
    gs = zeros(numcell * npts, ncol)
    xe = Coords_by_Ele(xc, conn)   
    count = 0
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
        error("Only 2D or 3D supported")
    end

    return gs
end

function egauss_bound(xc::Array{Float64,2}, conn::Array{Int,2}, gauss::Array{Float64,2})
    dim = size(xc, 2)
    l = size(gauss, 2)
    npts = l^(dim-1)
    numcell = size(conn, 1)
    ncol = dim + 2
    gs = zeros(numcell * npts, ncol)
    xe = Coords_by_Ele(xc, conn)
    count = 0
    if dim == 2
            for i in 1:l
                count += 1
                xi = gauss[1, i]
                xg = zeros(numcell, 2)
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
        error("Only 2D or 3D supported")
    end

    return gs
end

function fem_shape_line2(xi)
    phi = [0.5 * (1 - xi), 0.5 * (1 + xi)]
    dphi_l = [-0.5  0.5]
    return phi, dphi_l
end

# 2D: Para volumen en 2D o fronteras en 3D (Quad de 4 nodos)
function fem_shape_quad4(xi, eta)
    phi = [
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ]
    dphi_l = [
        -0.25*(1-eta)  0.25*(1-eta) 0.25*(1+eta) -0.25*(1+eta);
        -0.25*(1-xi)  -0.25*(1+xi)  0.25*(1+xi)  0.25*(1-xi)
    ]
    return phi, dphi_l
end

# 3D: Para volumen en 3D (Hexaedro de 8 nodos)
function fem_shape_hex8(xi, eta, zeta)
    phi = [
        (1/8)*(1-xi)*(1-eta)*(1-zeta), (1/8)*(1+xi)*(1-eta)*(1-zeta),
        (1/8)*(1+xi)*(1+eta)*(1-zeta), (1/8)*(1-xi)*(1+eta)*(1-zeta),
        (1/8)*(1-xi)*(1-eta)*(1+zeta), (1/8)*(1+xi)*(1-eta)*(1+zeta),
        (1/8)*(1+xi)*(1+eta)*(1+zeta), (1/8)*(1-xi)*(1+eta)*(1+zeta)
    ]
    dphi_l = zeros(3, 8)
    dphi_l[1,:] = [-(1/8)*(1-eta)*(1-zeta), (1/8)*(1-eta)*(1-zeta), (1/8)*(1+eta)*(1-zeta), -(1/8)*(1+eta)*(1-zeta), 
                   -(1/8)*(1-eta)*(1+zeta), (1/8)*(1-eta)*(1+zeta), (1/8)*(1+eta)*(1+zeta), -(1/8)*(1+eta)*(1+zeta)]
    dphi_l[2,:] = [-(1/8)*(1-xi)*(1-zeta), -(1/8)*(1+xi)*(1-zeta), (1/8)*(1+xi)*(1-zeta), (1/8)*(1-xi)*(1-zeta), 
                   -(1/8)*(1-xi)*(1+zeta), -(1/8)*(1+xi)*(1+zeta), (1/8)*(1+xi)*(1+zeta), (1/8)*(1-xi)*(1+zeta)]
    # d/dzeta
    dphi_l[3,:] = [-(1/8)*(1-xi)*(1-eta), -(1/8)*(1+xi)*(1-eta), -(1/8)*(1+xi)*(1+eta), -(1/8)*(1-xi)*(1+eta), 
                    (1/8)*(1-xi)*(1-eta),  (1/8)*(1+xi)*(1-eta),  (1/8)*(1+xi)*(1+eta),  (1/8)*(1-xi)*(1+eta)]
    return phi, dphi_l
end

# ==============================================================================
# INTEGRACIÓN DE VOLUMEN (DOMINIO)
# ==============================================================================

function egauss_fem(xc::Matrix{Float64}, conn::Matrix{Int}, gauss_rule::Matrix{Float64})
    dim = size(xc, 2)
    l = size(gauss_rule, 2)
    npts_cell = l^dim
    numcell = size(conn, 1)
    numv_by_cell= size(conn,2)
    total_gauss = numcell * npts_cell
    # Estructuras idénticas a IMLS
    gs = zeros(total_gauss, dim + 2)
    PHI  = Vector{Vector{Float64}}(undef, total_gauss)
    DPHI = Vector{Vector{SVector{dim, Float64}}}(undef, total_gauss)
    DOM  = Vector{Vector{Int}}(undef, total_gauss)
    xe_all = Coords_by_Ele(xc, conn)
    idx = 1
    if dim == 2
        @inbounds for e in 1:numcell
            xe = @view xe_all[e, :, :]
            for j in 1:l, i in 1:l
                xi, eta = gauss_rule[1, i], gauss_rule[1, j]
                w = gauss_rule[2, i] * gauss_rule[2, j]
                
                phi, dphi_l = fem_shape_quad4(xi, eta)
                xg = SVector{2}(sum(phi[k] * xe[k, 1] for k in 1:numv_by_cell), sum(phi[k] * xe[k, 2] for k in 1:numv_by_cell))
                J = @SMatrix [
                    sum(dphi_l[1, k] * xe[k, 1] for k in 1:numv_by_cell)  sum(dphi_l[1, k] * xe[k, 2] for k in 1:numv_by_cell);
                    sum(dphi_l[2, k] * xe[k, 1] for k in 1:numv_by_cell)  sum(dphi_l[2, k] * xe[k, 2] for k in 1:numv_by_cell)
                ]
                detJ = det(J)
                invJT = inv(J)'
                gs[idx, 1:2] .= xg; gs[idx, 3] = w; gs[idx, 4] = detJ
                PHI[idx] = phi
                DPHI[idx] = [invJT * SVector{2}(dphi_l[1, k], dphi_l[2, k]) for k in 1:numv_by_cell]
                DOM[idx] = conn[e, :]
                idx += 1
            end
        end
    elseif dim == 3
        @inbounds for e in 1:numcell
            xe = @view xe_all[e, :, :]
            for k in 1:l, j in 1:l, i in 1:l
                xi, eta, zeta = gauss_rule[1, i], gauss_rule[1, j], gauss_rule[1, k]
                w = gauss_rule[2, i] * gauss_rule[2, j] * gauss_rule[2, k]
                phi, dphi_l = fem_shape_hex8(xi, eta, zeta)
                xg = SVector{3}(sum(phi[m] * xe[m, 1] for m in 1:numv_by_cell), sum(phi[m] * xe[m, 2] for m in 1:numv_by_cell), sum(phi[m] * xe[m, 3] for m in 1:numv_by_cell))
                J = @SMatrix [
                    sum(dphi_l[1,m]*xe[m,1] for m=1:numv_by_cell) sum(dphi_l[1,m]*xe[m,2] for m=1:numv_by_cell) sum(dphi_l[1,m]*xe[m,3] for m=1:numv_by_cell);
                    sum(dphi_l[2,m]*xe[m,1] for m=1:numv_by_cell) sum(dphi_l[2,m]*xe[m,2] for m=1:numv_by_cell) sum(dphi_l[2,m]*xe[m,3] for m=1:numv_by_cell);
                    sum(dphi_l[3,m]*xe[m,1] for m=1:numv_by_cell) sum(dphi_l[3,m]*xe[m,2] for m=1:numv_by_cell) sum(dphi_l[3,m]*xe[m,3] for m=1:numv_by_cell)
                ]
                detJ = det(J)
                invJT = inv(J)'
                gs[idx, 1:3] .= xg; gs[idx, 4] = w; gs[idx, 5] = detJ
                PHI[idx] = phi
                DPHI[idx] = [invJT * SVector{3}(dphi_l[1, m], dphi_l[2, m], dphi_l[3, m]) for m in 1:numv_by_cell]
                DOM[idx] = conn[e, :]
                idx += 1
            end
        end
    end
    return gs, PHI, DPHI, DOM
end

function egauss_bound_fem(xc::Matrix{Float64}, conn::Matrix{Int}, gauss_rule::Matrix{Float64})
    dim = size(xc, 2)
    l = size(gauss_rule, 2)
    npts_bound = l^(dim-1)
    numcell = size(conn, 1)
    total_gauss = numcell * npts_bound
    gs = zeros(total_gauss, dim + 2)
    PHI  = Vector{Vector{Float64}}(undef, total_gauss)
    DPHI = Vector{Vector{SVector{dim, Float64}}}(undef, total_gauss)
    DOM  = Vector{Vector{Int}}(undef, total_gauss)
    xe_all = Coords_by_Ele(xc, conn)
    numv_by_cell= size(conn,2)
    idx = 1
    if dim == 2
       @inbounds for e in 1:numcell
            xe = @view xe_all[e, :, :]
            for i in 1:l
                xi = gauss_rule[1, i]; w = gauss_rule[2, i]
                phi, dphi_l = fem_shape_line2(xi)
                xg = SVector{2}(sum(phi[k] * xe[k, 1] for k in 1:numv_by_cell), sum(phi[k] * xe[k, 2] for k in 1:numv_by_cell))
                tx = sum(dphi_l[1, k] * xe[k, 1] for k in 1:2)
                ty = sum(dphi_l[1, k] * xe[k, 2] for k in 1:2)
                ds = sqrt(tx^2 + ty^2)
                gs[idx, 1:2] .= xg; gs[idx, 3] = w; gs[idx, 4] = ds
                PHI[idx] = phi
                DPHI[idx] = [SVector{2}(dphi_l[1,k]*tx / ds^2, dphi_l[1,k]*ty / ds^2) for k in 1:numv_by_cell]
                DOM[idx] = conn[e, :]
                idx += 1
            end
        end
    elseif dim == 3
        @inbounds for e in 1:numcell
            xe = @view xe_all[e, :, :]
            for j in 1:l, i in 1:l
                xi, eta = gauss_rule[1, i], gauss_rule[1, j]
                w = gauss_rule[2, i] * gauss_rule[2, j]
                phi, dphi_l = fem_shape_quad4(xi, eta)
                xg = SVector{3}(sum(phi[k]*xe[k,1] for k=1:numv_by_cell), sum(phi[k]*xe[k,2] for k=1:numv_by_cell), sum(phi[k]*xe[k,3] for k=1:numv_by_cell))
                t1 = SVector{3}(sum(dphi_l[1,k]*xe[k,1] for k=1:numv_by_cell), sum(dphi_l[1,k]*xe[k,2] for k=1:numv_by_cell), sum(dphi_l[1,k]*xe[k,3] for k=1:numv_by_cell))
                t2 = SVector{3}(sum(dphi_l[2,k]*xe[k,1] for k=1:numv_by_cell), sum(dphi_l[2,k]*xe[k,2] for k=1:numv_by_cell), sum(dphi_l[2,k]*xe[k,3] for k=1:numv_by_cell))
                g11, g12, g22 = dot(t1,t1), dot(t1,t2), dot(t2,t2)
                detG = g11*g22 - g12^2
                dA = sqrt(detG)
                invG = @SMatrix [g22/detG -g12/detG; -g12/detG g11/detG]
                gs[idx, 1:3] .= xg; gs[idx, 4] = w; gs[idx, 5] = dA
                PHI[idx] = phi
                DPHI_e = Vector{SVector{3, Float64}}(undef, numv_by_cell)
                for k in 1:numv_by_cell
                    dphi_loc = SVector{2}(dphi_l[1,k], dphi_l[2,k])
                    coeffs = invG * dphi_loc
                    DPHI_e[k] = coeffs[1]*t1 + coeffs[2]*t2
                end
                DPHI[idx] = DPHI_e
                DOM[idx] = conn[e, :]
                idx += 1
            end
        end
    end
    return gs, PHI, DPHI, DOM
end
