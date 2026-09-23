using LinearAlgebra
using StaticArrays

function domain(gpos::AbstractVector{T}, x::Matrix{Float64}, dm::Union{AbstractMatrix{T},AbstractVector{T}}; shape::Symbol=:rectangular) where T<:Real
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

function Improved_Moving_Least_Squares(gpos, x, v, dm; shape::Symbol=:rectangular, Layout::Symbol=:multilinear)
    L = length(v)
    dim = length(gpos)
    nv = @view x[v, :]
    is_linear = (Layout === :linear)
    if is_linear
        npoly = (dim == 2) ? 3 : (dim == 3 ? 4 : error("Unsupported dimension: $dim"))
    else
        npoly = (dim == 2) ? 4 : (dim == 3 ? 8 : error("Unsupported dimension: $dim"))
    end
    q = Array{Float64}(undef, L, npoly)
    if is_linear
        if dim == 2
            @inbounds for i in 1:L
                q[i, 1] = 1.0
                q[i, 2] = nv[i, 1]
                q[i, 3] = nv[i, 2]
            end
            QQg = [1.0 gpos[1] gpos[2];
                   0.0 1.0 0.0;
                   0.0 0.0 1.0]
        elseif dim == 3
            @inbounds for i in 1:L
                q[i, 1] = 1.0
                q[i, 2] = nv[i, 1]
                q[i, 3] = nv[i, 2]
                q[i, 4] = nv[i, 3]
            end
            QQg = [1.0 gpos[1] gpos[2] gpos[3];
                   0.0 1.0 0.0 0.0;
                   0.0 0.0 1.0 0.0;
                   0.0 0.0 0.0 1.0]
        end
    else
        if dim == 2
            @inbounds for i in 1:L
                x_, y_ = nv[i, 1], nv[i, 2]
                q[i, 1] = 1.0
                q[i, 2] = x_
                q[i, 3] = y_
                q[i, 4] = x_ * y_
            end
            QQg = [1.0 gpos[1] gpos[2] gpos[1]*gpos[2];
                   0.0 1.0 0.0 gpos[2];
                   0.0 0.0 1.0 gpos[1]]
        elseif dim == 3
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
            QQg = [1.0 gpos[1] gpos[2] gpos[3] gpos[1]*gpos[2] gpos[1]*gpos[3] gpos[2]*gpos[3] gpos[1]*gpos[2]*gpos[3];
                   0.0 1.0 0.0 0.0 gpos[2] gpos[3] 0.0 gpos[2]*gpos[3];
                   0.0 0.0 1.0 0.0 gpos[1] 0.0 gpos[3] gpos[1]*gpos[3];
                   0.0 0.0 0.0 1.0 0.0 gpos[1] gpos[2] gpos[1]*gpos[2]]
        end
    end
    dif = Array{Float64}(undef, L, dim)
    @inbounds for i in 1:L, d in 1:dim
        dif[i, d] = gpos[d] - nv[i, d]
    end
    w, dw = cubwgt(dif, v, dm; shape=shape)
    p = copy(q)
    PPg = copy(QQg)
    @inbounds for i in 2:npoly
        for k in 1:(i-1)
            num = 0.0
            den = 0.0
            for j in 1:L
                wj = w[j]
                pk_j = p[j, k]
                num += q[j, i] * wj * pk_j
                den += pk_j * wj * pk_j
            end
            c1 = num / den
            
            for j in 1:L
                p[j, i] -= c1 * p[j, k]
            end
            for r in axes(PPg, 1)
                PPg[r, i] -= c1 * PPg[r, k]
            end
        end
    end
    aia = zeros(npoly)
    @inbounds for k in 1:npoly
        den = 0.0
        for j in 1:L
            den += p[j, k] * w[j] * p[j, k]
        end
        aia[k] = 1.0 / den
    end
    gam = zeros(npoly, dim + 1)
    @inbounds for k in 1:npoly
        gam[k, 1] = aia[k] * PPg[1, k]
    end
    daa = Array{Float64}(undef, npoly, npoly)
    @inbounds for d in 1:dim
        fill!(daa, 0.0)
        dw_d = @view dw[:, d]
        for c2 in 1:npoly
            for c1 in 1:npoly
                s = 0.0
                for j in 1:L
                    s += p[j, c1] * dw_d[j] * p[j, c2]
                end
                daa[c1, c2] = s
            end
        end 
        for k in 1:npoly
            s_daa_gam = 0.0
            for c in 1:npoly
                s_daa_gam += daa[k, c] * gam[c, 1]
            end
            gam[k, d+1] = aia[k] * (PPg[d+1, k] - s_daa_gam)
        end
    end
    phi = zeros(L)
    dphi_x = zeros(L)
    dphi_y = zeros(L)
    dphi_z = (dim == 3) ? zeros(L) : nothing
    @inbounds for j in 1:L
        wj = w[j]
        dwj_1 = dw[j, 1]
        dwj_2 = dw[j, 2]
        dwj_3 = (dim == 3) ? dw[j, 3] : 0.0
        s_phi = 0.0
        s_dx = 0.0
        s_dy = 0.0
        s_dz = 0.0
        for c in 1:npoly
            pcj = p[j, c]
            g1 = gam[c, 1]
            g2 = gam[c, 2]
            g3 = gam[c, 3] 
            s_phi += pcj * wj * g1
            s_dx += (pcj * dwj_1 * g1) + (pcj * wj * g2)
            s_dy += (pcj * dwj_2 * g1) + (pcj * wj * g3)
            if dim == 3
                g4 = gam[c, 4]
                s_dz += (pcj * dwj_3 * g1) + (pcj * wj * g4)
            end
        end 
        phi[j] = s_phi
        dphi_x[j] = s_dx
        dphi_y[j] = s_dy
        if dim == 3
            dphi_z[j] = s_dz
        end
    end
    if dim == 2
        dphi = [SVector{2, Float64}(dphi_x[a], dphi_y[a]) for a in 1:L]
    else
        dphi = [SVector{3, Float64}(dphi_x[a], dphi_y[a], dphi_z[a]) for a in 1:L]
    end
    return phi, dphi
end

function Moving_Kriging(gpos, x, v, dm; shape::Symbol=:rectangular, Cq::Float64=4.3, Layout::Symbol=:multilinear)
    L = length(v)
    dim = length(gpos)
    theta = 9.0 * (Cq^2)
    xye = [sum(x[v, d])/L for d in 1:dim]
    h_avg = (shape === :rectangular) ? [sum(dm[v, d])/L for d in 1:dim] : fill(sum(dm[v])/L, dim)
    den_poly = h_avg .* Cq
    is_linear = (Layout === :linear)
    if dim == 2
        npoly = is_linear ? 3 : 4
        P = ones(L, npoly)
        p = ones(npoly)
        dp = zeros(npoly, dim)
        gn = [(gpos[d] - xye[d]) / den_poly[d] for d in 1:2]
        if is_linear
            for i in 1:L
                nx = (x[v[i], 1] - xye[1]) / den_poly[1]
                ny = (x[v[i], 2] - xye[2]) / den_poly[2]
                P[i, 2], P[i, 3] = nx, ny
            end
            p[2], p[3] = gn[1], gn[2]
            dp[2, 1] = 1.0 / den_poly[1]
            dp[3, 2] = 1.0 / den_poly[2]
        else # :multilinear
            for i in 1:L
                nx = (x[v[i], 1] - xye[1]) / den_poly[1]
                ny = (x[v[i], 2] - xye[2]) / den_poly[2]
                P[i, 2], P[i, 3], P[i, 4] = nx, ny, nx * ny
            end
            p[2], p[3], p[4] = gn[1], gn[2], gn[1] * gn[2]
            dp[2, 1] = 1.0 / den_poly[1];
            dp[4, 1] = gn[2] / den_poly[1]
            dp[3, 2] = 1.0 / den_poly[2];
            dp[4, 2] = gn[1] / den_poly[2]
        end
    else # dim == 3
        npoly = is_linear ? 4 : 8
        P = ones(L, npoly)
        p = ones(npoly)
        dp = zeros(npoly, dim)
        gn = [(gpos[d] - xye[d]) / den_poly[d] for d in 1:3]
        if is_linear
            for i in 1:L
                nx, ny, nz = [(x[v[i], d] - xye[d]) / den_poly[d] for d in 1:3]
                P[i, 2], P[i, 3], P[i, 4] = nx, ny, nz
            end
            p[2], p[3], p[4] = gn[1], gn[2], gn[3]
            dp[2, 1] = 1.0 / den_poly[1]
            dp[3, 2] = 1.0 / den_poly[2]
            dp[4, 3] = 1.0 / den_poly[3]
        else # :multilinear
            for i in 1:L
                nx, ny, nz = [(x[v[i], d] - xye[d]) / den_poly[d] for d in 1:3]
                P[i, 2], P[i, 3], P[i, 4] = nx, ny, nz
                P[i, 5], P[i, 6], P[i, 7] = nx*ny, nx*nz, ny*nz
                P[i, 8] = nx*ny*nz
            end
            p[2], p[3], p[4] = gn[1], gn[2], gn[3]
            p[5], p[6], p[7] = gn[1]*gn[2], gn[1]*gn[3], gn[2]*gn[3]
            p[8] = gn[1]*gn[2]*gn[3]
            dp[2, 1] = 1.0/den_poly[1];
            dp[5, 1] = gn[2]/den_poly[1];
            dp[6, 1] = gn[3]/den_poly[1];
            dp[8, 1] = gn[2]*gn[3]/den_poly[1]
            dp[3, 2] = 1.0/den_poly[2];
            dp[5, 2] = gn[1]/den_poly[2];
            dp[7, 2] = gn[3]/den_poly[2];
            dp[8, 2] = gn[1]*gn[3]/den_poly[2]
            dp[4, 3] = 1.0/den_poly[3];
            dp[6, 3] = gn[1]/den_poly[3];
            dp[7, 3] = gn[2]/den_poly[3];
            dp[8, 3] = gn[1]*gn[2]/den_poly[3]
        end
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
            den = den_poly[d]
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

function safe_inv_hessian(H::SMatrix{dim, dim, Float64, N}, rtol::Float64=1e-7) where {dim, N}
    # Descomposición espectral analítica/optimizada para SMatrix simétrica
    E = eigen(Symmetric(H))
    max_ev = maximum(abs, E.values)
    
    # Si la matriz colapsó por completo (coincidencia nodal exactísima)
    if max_ev < 1e-10
        return zeros(SMatrix{dim, dim, Float64, N})
    end
    
    # Invertir únicamente los autovalores mayores al tolerancia relativa
    inv_vals = MVector{dim, Float64}(undef)
    for d in 1:dim
        if abs(E.values[d]) > rtol * max_ev
            inv_vals[d] = 1.0 / E.values[d]
        else
            inv_vals[d] = 0.0  # Filtra direcciones colineales o numéricamente nulas
        end
    end
    
    V = E.vectors
    return V * Diagonal(SVector(inv_vals)) * V'
end

function Local_Maximum_Entropy(gpos::AbstractVector{T}, x::Matrix{Float64}, v::Vector{Int}, dm; 
                               gamma::Float64=1.0, shape::Symbol=:rectangular, max_iter::Int=30, tol::Float64=1e-10) where T<:Real
    L = length(v)
    dim = length(gpos) 
    
    beta = MVector{dim, Float64}(undef)
    if shape === :rectangular
        for d in 1:dim
            h_avg_d = (dm isa AbstractMatrix) ? sum(dm[v, d])/L : sum(dm[v])/L
            beta[d] = gamma / (h_avg_d^2)
        end
    else
        h_avg_val = (dm isa AbstractMatrix) ? sum(dm[v, 1])/L : sum(dm[v])/L
        b_val = gamma / (h_avg_val^2)
        fill!(beta, b_val)
    end

    dx = Array{Float64}(undef, L, dim)
    @inbounds for i in 1:L, d in 1:dim
        dx[i, d] = x[v[i], d] - gpos[d]
    end

    lambda = zeros(SVector{dim, Float64})
    p = zeros(L)
    H = zeros(SMatrix{dim, dim, Float64})
    
    # Solver Newton-Raphson
    for iter in 1:max_iter
        z_sum = 0.0
        @inbounds for i in 1:L
            exp_prior = 0.0
            dot_l_dx = 0.0
            for d in 1:dim
                exp_prior -= beta[d] * (dx[i, d]^2)
                dot_l_dx += lambda[d] * dx[i, d]
            end
            arg = clamp(exp_prior - dot_l_dx, -700.0, 700.0)
            w_i = exp(arg)
            p[i] = w_i
            z_sum += w_i
        end
        
        p ./= z_sum
        
        r_temp = MVector{dim, Float64}(zeros(dim))
        @inbounds for i in 1:L, d in 1:dim
            r_temp[d] -= p[i] * dx[i, d]
        end
        r = SVector(r_temp)
        
        H_temp = MMatrix{dim, dim, Float64}(zeros(dim, dim))
        @inbounds for i in 1:L
            dxi = SVector{dim, Float64}(@view dx[i, :])
            H_temp .+= p[i] .* (dxi * dxi')
        end
        H = SMatrix(H_temp) - (r * r')
        
        if norm(r) < tol
            break
        end
        
        # Inversión segura también dentro del paso de Newton
        invH_step = safe_inv_hessian(H)
        d_lambda = invH_step * r
        lambda -= d_lambda
        
        if norm(d_lambda) < tol
            break
        end
    end

    # Inversión segura para el cálculo analítico de gradientes (sin 1e-12 * I)
    invH = safe_inv_hessian(H)
    
    dphi_raw = Array{Float64}(undef, L, dim)
    @inbounds for i in 1:L
        dxi = SVector{dim, Float64}(@view dx[i, :])
        # Gradiente analítico exacto de LME: grad_i = p_i * H^-1 * dx_i
        grad_i = p[i] .* (invH * dxi)
        for d in 1:dim
            dphi_raw[i, d] = grad_i[d]
        end
    end

    if dim == 2
        dphi = [SVector{2, Float64}(dphi_raw[a, 1], dphi_raw[a, 2]) for a in 1:L]
    else
        dphi = [SVector{3, Float64}(dphi_raw[a, 1], dphi_raw[a, 2], dphi_raw[a, 3]) for a in 1:L]
    end 
    return p, dphi
end

function SHAPE_FUN(gs, x, dm; shape::Symbol=:rectangular, technique::Symbol=:IMLS, Layout::Symbol=:multilinear, gamma::Float64=5.0, Cq::Float64=4.3)
    numqc = size(gs, 1)
    dim = size(x, 2)
    PHI = Vector{Vector{Float64}}(undef, numqc)
    DPHI = Vector{Vector{SVector{dim,Float64}}}(undef, numqc)
    DOM = Vector{Vector{Int}}(undef, numqc)
    @inbounds for ind in 1:numqc
        gpos = SVector{dim,Float64}(gs[ind, 1:dim])
        v = domain(gpos, x, dm; shape=shape)
        if technique === :IMLS
            phi, dphi = Improved_Moving_Least_Squares(gpos, x, v, dm; shape=shape, Layout=Layout)
        elseif technique === :MK
            phi, dphi = Moving_Kriging(gpos, x, v, dm; shape=shape, Layout=Layout,Cq=Cq)
        elseif technique === :LME
            phi, dphi = Local_Maximum_Entropy(gpos, x, v, dm; gamma=gamma, shape=shape)
        else
            error("Unknown technique: $technique. Supported: IMLS, MK, LME.")
        end     
        PHI[ind] = phi
        DPHI[ind] = dphi
        DOM[ind] = v
    end
    return PHI, DPHI, DOM
end