using LinearAlgebra

function domain(gpos::AbstractVector{T}, x::AbstractMatrix{T}, dm::Union{AbstractMatrix{T},AbstractVector{T}}; shape::Symbol=:rectangular) where T<:Real
    nnodes, dim = size(x)
    inside = trues(nnodes)  # vector booleano pre-alocado
    if shape === :rectangular
        for j in 1:dim
            colx = @view x[:, j]
            coldm = @view dm[:, j]
            g = gpos[j]
            @inbounds for i in 1:nnodes
                inside[i] &= abs(colx[i] - g) <= coldm[i]
            end
        end
    elseif shape === :circular || shape === :spherical
        @inbounds for i in 1:nnodes
            dist_sq = zero(T)
            for j in 1:dim
                dist_sq += (x[i, j] - gpos[j])^2
            end
            inside[i] = dist_sq <= (dm[i])^2
        end
    else
        error("Unknown shape of influence domain: $shape. Shapes supported are: rectangular, circular, spherical, cylindrical.")
    end
    return findall(inside)
end

function cubwgt(dif, v, dm; shape::Symbol=:rectangular)
    l, dim = size(dif)  # dim = 2 o 3
    w = zeros(l)
    dw = zeros(l, dim)
    if shape === :rectangular
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
    elseif shape === :circular || shape === :spherical
        for i in 1:l
            dmi = dm[v[i]]
            dist = 0.0
            for d in 1:dim
                dist += dif[i, d]^2
            end
            dist = sqrt(dist)
            r = dist / dmi
            if r <= 1.0
                if r <= 0.5
                    f = (2 / 3) - 4 * r^2 + 4 * r^3
                    df_dr = -8 * r + 12 * r^2
                else
                    f = (4 / 3) - 4 * r + 4 * r^2 - (4 / 3) * r^3
                    df_dr = -4 + 8 * r - 4 * r^2
                end
                w[i] = f
                if dist > 1e-12
                    for d in 1:dim
                        dw[i, d] = df_dr * (1 / dmi) * (dif[i, d] / dist)
                    end
                end
            end
        end
    end
    return w, dw
end

function Improved_Moving_Least_Squares(gpos, x, v, dm; shape::Symbol=:rectangular)
    L = length(v)
    dim = length(gpos)
    nv = x[v, :]
    if dim == 2
        q = Array{Float64}(undef, L, 4)
        @inbounds for i in 1:L
            q[i, 1] = 1.0
            q[i, 2] = nv[i, 1]
            q[i, 3] = nv[i, 2]
            q[i, 4] = nv[i, 1] * nv[i, 2]
        end
    elseif dim == 3
        q = Array{Float64}(undef, L, 8)
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
        error("Only 2D or 3D problems are supported. Detected dimension: $dim.")
    end
    dif = Array{Float64}(undef, L, dim)
    @inbounds for i in 1:L
        for d in 1:dim
            dif[i, d] = gpos[d] - nv[i, d]
        end
    end
    w, dw = cubwgt(dif, v, dm; shape=shape)
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
    gam = zeros(npoly, dim + 1)
    gam[:, 1] = aia .* PPg[1, :]
    for d in 1:dim
        daa = p' * (dw[:, d] .* p)
        gam[:, d+1] = aia .* (PPg[d+1, :] - daa * gam[:, 1])
    end
    B = p .* w
    phi = (B * gam[:, 1])
    dphi_x = (B * gam[:, 2]) + ((p .* dw[:, 1]) * gam[:, 1])
    dphi_y = (B * gam[:, 3]) + ((p .* dw[:, 2]) * gam[:, 1])
    if dim == 2
        dphi = [SVector{2,Float64}(dphi_x[a], dphi_y[a]) for a in 1:L]
    else
        dphi_z = (B * gam[:, 4]) + ((p .* dw[:, 3]) * gam[:, 1])
        dphi = [SVector{3,Float64}(dphi_x[a], dphi_y[a], dphi_z[a]) for a in 1:L]
    end
    return phi, dphi
end

function Moving_Kriging(gpos, x, v, dm; shape::Symbol=:rectangular, Cq::Float64=4.3)
    L = length(v)
    dim = length(gpos)
    theta = 0.1 * (Cq^2)
    xye = [sum(x[v, d])/L for d in 1:dim]
    h_avg = (shape === :rectangular) ? [sum(dm[v, d])/L for d in 1:dim] : fill(sum(dm[v])/L, dim)
    den_poly = h_avg .* Cq
    npoly = (dim == 2) ? 4 : 8
    P = ones(L, npoly)
    p = ones(npoly)
    dp = zeros(npoly, dim)
    gn = [(gpos[d] - xye[d]) / den_poly[d] for d in 1:dim]
    if dim == 2
        for i in 1:L
            nx = (x[v[i], 1] - xye[1]) / den_poly[1]
            ny = (x[v[i], 2] - xye[2]) / den_poly[2]
            P[i, 2], P[i, 3], P[i, 4] = nx, ny, nx * ny
        end
        p[2], p[3], p[4] = gn[1], gn[2], gn[1] * gn[2]
        dp[2, 1] = 1/den_poly[1]; dp[4, 1] = gn[2]/den_poly[1]
        dp[3, 2] = 1/den_poly[2]; dp[4, 2] = gn[1]/den_poly[2]
    else
        for i in 1:L
            nx, ny, nz = [(x[v[i], d] - xye[d]) / den_poly[d] for d in 1:dim]
            P[i, 2], P[i, 3], P[i, 4] = nx, ny, nz
            P[i, 5], P[i, 6], P[i, 7] = nx*ny, nx*nz, ny*nz
            P[i, 8] = nx*ny*nz
        end
        p[2], p[3], p[4] = gn[1], gn[2], gn[3]
        p[5], p[6], p[7] = gn[1]*gn[2], gn[1]*gn[3], gn[2]*gn[3]
        p[8] = gn[1]*gn[2]*gn[3]
        dp[2, 1] = 1/den_poly[1]; dp[5, 1] = gn[2]/den_poly[1]; dp[6, 1] = gn[3]/den_poly[1]; dp[8, 1] = gn[2]*gn[3]/den_poly[1]
        dp[3, 2] = 1/den_poly[2]; dp[5, 2] = gn[1]/den_poly[2]; dp[7, 2] = gn[3]/den_poly[2]; dp[8, 2] = gn[1]*gn[3]/den_poly[2]
        dp[4, 3] = 1/den_poly[3]; dp[6, 3] = gn[1]/den_poly[3]; dp[7, 3] = gn[2]/den_poly[3]; dp[8, 3] = gn[1]*gn[2]/den_poly[3]
    end
    Rij = zeros(L, L)
    for i in 1:L, j in 1:L
        arg = 0.0
        for d in 1:dim
            den = den_poly[d]
            dist_d = (x[v[i], d] - x[v[j], d]) / den
            arg += dist_d^2
        end
        Rij[i, j] = exp(-theta * arg)
    end
    r = zeros(L)
    dr = zeros(L, dim)
    for i in 1:L
        arg = 0.0
        diff_scaled = zeros(dim)
        for d in 1:dim
            den =den_poly[d]
            diff_scaled[d] = (gpos[d] - x[v[i], d]) / den
            arg += diff_scaled[d]^2
        end
        val_r = exp(-theta * arg)
        r[i] = val_r
        for d in 1:dim
            den = den_poly[d]
            dr[i, d] = val_r * (-2.0 * theta * diff_scaled[d] / den)
        end
    end
    LU_R = lu(Rij)
    R1P = LU_R \ P
    M = P' * R1P
    A = M \ (R1P')
    B = LU_R \ (I(L) - P * A)
    phi = vec(p' * A + r' * B)
    dphi_raw = [dp[:, d]' * A + dr[:, d]' * B for d in 1:dim]
    if dim == 2
        dphi = [SVector{2,Float64}(dphi_raw[1][i], dphi_raw[2][i]) for i in 1:L]
    else
        dphi = [SVector{3,Float64}(dphi_raw[1][i], dphi_raw[2][i], dphi_raw[3][i]) for i in 1:L]
    end
    return phi, dphi
end

function SHAPE_FUN(gs, x, dm; shape::Symbol=:rectangular, technique::Symbol=:IMLS)
    numqc = size(gs, 1)
    dim = size(x, 2)
    PHI = Vector{Vector{Float64}}(undef, numqc)
    # El tipo ahora es explícito: Vector de Vectores de SVectors
    DPHI = Vector{Vector{SVector{dim,Float64}}}(undef, numqc)
    DOM = Vector{Vector{Int}}(undef, numqc)
    @inbounds for ind in 1:numqc
        gpos = SVector{dim,Float64}(gs[ind, 1:dim]) # gpos como SVector ayuda mucho
        v = domain(gpos, x, dm; shape=shape)
        if technique === :IMLS
        phi, dphi = Improved_Moving_Least_Squares(gpos, x, v, dm; shape=shape)
        elseif technique === :MK
        phi, dphi = Moving_Kriging(gpos, x, v, dm; shape=shape)
        else
        error("Unknown technique: $technique. Techniques supported are: IMLS, MK. If you wrote MLS, please write IMLS instead, which is more efficient than standard MLS providing the same results.")
        end
        PHI[ind] = phi
        DPHI[ind] = dphi
        DOM[ind] = v
    end
    return PHI, DPHI, DOM
end
