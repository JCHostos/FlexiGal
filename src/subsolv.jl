function subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a_mma, bm, c, d)
    # This is the primal-dual algorithm for solving the MMA subproblem
    # Initialisation
    een = ones(n)
    eem = ones(m)
    epsi = 1.0
    x = 0.5 .* (alfa .+ beta)  #Fulfiling x-α>0 and x-β<0
    y = copy(eem)  # Fulfiling y>0
    z = 1.0 # Fulfiling z>0
    lam = copy(eem) # Fulfiling λ>0
    xsi = een ./ (x .- alfa) # Fulfiling ξ>0
    xsi = max.(xsi, een) # Choosing the maximum possible initialisation
    eta = een ./ (beta .- x) # Fulgilling η>0
    eta = max.(eta, een) # Choosing the maximum possible initialisation
    mu = max.(eem, 0.5 .* c) # Fulfiling μ>0 and commonly c>>>1
    zet = 1.0 # Fulfiling ζ>0
    s = copy(eem) # Fulfiling s>0
    itera = 0

    while epsi > epsimin
        epsvecn = epsi .* een
        epsvecm = epsi .* eem
        ux1 = upp .- x # Denominator for p in g
        xl1 = x .- low # Denominator for q in g
        ux2 = ux1 .^ 2 # Denominator for p in dψdx
        xl2 = xl1 .^ 2 # Denominator for q in dψdx
        uxinv1 = een ./ ux1 # Inverse for p in g 
        xlinv1 = een ./ xl1 # Inverse for q in g
        plam = p0 .+ P' * lam # Augmented Lagrangian for p
        qlam = q0 .+ Q' * lam # Augmented Lagrangian for q
        gvec = P * uxinv1 .+ Q * xlinv1 # Vector for each constraint
        dpsidx = plam ./ ux2 .- qlam ./ xl2 # Derivative of auxiliar function ψ for Augmented Lagrangian
        # The optimality consists of approaching the residuals to zero
        rex = dpsidx .- xsi .+ eta # Residual for x
        rey = c .+ d .* y .- mu .- lam # Residual for y
        rez = a0 .- zet .- a_mma' * lam # Residual for ζ
        relam = gvec .- a_mma * z .- y .+ s .- bm # Residual for Lagrange multiplier
        rexsi = xsi .* (x .- alfa) .- epsvecn # Residual for ξ
        reeta = eta .* (beta .- x) .- epsvecn # Residual for η
        remu = mu .* y .- epsvecm # Residual for μ
        rezet = zet * z .- epsi # Residual for z 
        res = lam .* s .- epsvecm # Residual for s
        residu1 = vcat(rex, rey, rez) # Concatenating residuals for x, y and z
        residu2 = vcat(relam, rexsi, reeta, remu, rezet, res) # Concatenating residuals for λ, ξ, η, μ, ζ and s
        residu = vcat(residu1, residu2) # Concatenating all residuals in w
        residunorm = norm(residu) # Norm of the residual vector
        residumax = norm(residu, Inf) # Infinity norm of the residual vector
        ittt = 0

        while residumax > 0.9 * epsi && ittt < 200
            ittt += 1
            itera += 1
            # Tangent operator (¿Hessian?)
            ux1 = upp .- x # Updating denominators for p in g
            xl1 = x .- low # Updating denominators for q in g
            ux2 = ux1 .^ 2 # Updating denominators for p in dψdx
            xl2 = xl1 .^ 2 # Updating denominators for q in dψdx
            ux3 = ux1 .* ux2 # Denominator for p in dψ²dx² 
            xl3 = xl1 .* xl2 # Denominator for q in dψ²dx²
            uxinv1 = 1.0 ./ ux1 # Inverse for p in g
            xlinv1 = 1.0 ./ xl1 # Inverse for q in g
            uxinv2 = 1.0 ./ ux2 # Inverse for p in dψdx
            xlinv2 = 1.0 ./ xl2 # Inverse for q in dψdx
            plam = p0 .+ P' * lam # Augmented Lagrangian for p
            qlam = q0 .+ Q' * lam # Augmented Lagrangian for q
            gvec = P * uxinv1 .+ Q * xlinv1 # Vector for each constraint
            GG = P .* uxinv2' .- Q .* xlinv2' # Assembpling G
            dpsidx = plam ./ ux2 .- qlam ./ xl2 # First Derivative of auxiliar function ψ for Augmented Lagrangian
            # These are the modified residuals for the reduced system
            delx = dpsidx .- epsvecn ./ (x .- alfa) .+ epsvecn ./ (beta .- x) # Modified residual for x
            dely = c .+ d .* y .- lam .- epsvecm ./ y # Modified residual for y
            delz = a0 .- a_mma' * lam .- epsi / z # Modified residual for z
            dellam = gvec .- a_mma * z .- y .- bm .+ epsvecm ./ lam # Modified residual for λ
            diagx = plam./ux3 .+ qlam./xl3;
            diagx = 2*diagx .+ xsi ./ (x .- alfa) .+ eta ./ (beta .- x) # Components of diagonal matrix Dₓ for reduced system
            diagxinv = een ./ diagx # Reciprocal of Dₓ
            diagy = d .+ mu ./ y # Components of diagonal matrix Dᵧ for reduced system
            diagyinv = eem ./ diagy # Reciprocal of Dᵧ
            diaglam = s ./ lam # Components of diagonal matrix Dλ for reduced system
            diaglamyi = diaglam .+ diagyinv # Components of diagonal matrix Dλy for reduced system
            if m < n # This is the most common case (Example: TopOpt with single volume constraint)
                blam = dellam .+ dely ./ diagy .- GG * (delx ./ diagx) # Load vector for Δλ
                bb = vcat(blam, delz) # Load vector for the reduced system (Contatenating bλ and δz)
                Alam = Diagonal(diaglamyi) + GG .* diagxinv' * GG' # 11 Block of the reduced system
                AA = [Alam a_mma; a_mma' -zet/z] # Assembling (m+1)x(m+1) Blocks reduced system
                solut = AA \ bb # Solving the reduced system
                dlam = solut[1:m] # Extracting Δλ
                dz = solut[m+1] # Extracting Δz
                dx = -delx ./ diagx .- (GG' * dlam) ./ diagx # Computing Δx
            else
                # Not important for now
                diaglamyiinv = eem ./ diaglamyi
                dellamyi = dellam .+ dely ./ diagy
                Axx = spdiagm(0 => diagx) + GG' .* diaglamyiinv' * GG
                azz = zet / z + a_mma' * (a_mma ./ diaglamyi)
                axz = -GG' * (a_mma ./ diaglamyi)
                bx = delx .+ GG' * (dellamyi ./ diaglamyi)
                bz = delz .- a_mma' * (dellamyi ./ diaglamyi)
                AA = [Axx axz; axz' azz]
                bb = -vcat(bx, bz)
                solut = AA \ bb
                dx = solut[1:n]
                dz = solut[n+1]
                dlam = (GG * dx) ./ diaglamyi .- dz .* (a_mma ./ diaglamyi) .+ dellamyi ./ diaglamyi
            end

            dy = -dely ./ diagy .+ dlam ./ diagy # Computing Δy
            dxsi = -xsi .+ epsvecn ./ (x .- alfa) .- (xsi .* dx) ./ (x .- alfa) # Computing Δξ
            deta = -eta .+ epsvecn ./ (beta .- x) .+ (eta .* dx) ./ (beta .- x) # Computing Δη
            dmu = -mu .+ epsvecm ./ y .- (mu .* dy) ./ y # Computing Δμ
            dzet = -zet .+ epsi / z .- zet * dz / z # Computing Δζ
            ds = -s .+ epsvecm ./ lam .- (s .* dlam) ./ lam  # Computing Δs

            xx = vcat(y, z, lam, xsi, eta, mu, zet, s)
            dxx = vcat(dy, dz, dlam, dxsi, deta, dmu, dzet, ds)
            stepxx = -1.01 .* dxx ./ xx
            stmxx = maximum(stepxx)
            stepalfa = -1.01 .* dx ./ (x .- alfa)
            stmalfa = maximum(stepalfa)
            stepbeta = 1.01 .* dx ./ (beta .- x)
            stmbeta = maximum(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            # Guardar estados
            xold = copy(x)
            yold = copy(y)
            zold = z
            lamold = copy(lam)
            xsiold = copy(xsi)
            etaold = copy(eta)
            muold = copy(mu)
            zetold = zet
            sold = copy(s)

            itto = 0
            resinew = 2 * residunorm

            while resinew > residunorm && itto < 50
                itto += 1
                x = xold .+ steg .* dx
                y = yold .+ steg .* dy
                z = zold + steg * dz
                lam = lamold .+ steg .* dlam
                xsi = xsiold .+ steg .* dxsi
                eta = etaold .+ steg .* deta
                mu = muold .+ steg .* dmu
                zet = zetold + steg * dzet
                s = sold .+ steg .* ds

                ux1 = upp .- x
                xl1 = x .- low
                ux2 = ux1 .^ 2
                xl2 = xl1 .^ 2
                uxinv1 = een ./ ux1
                xlinv1 = een ./ xl1
                plam = p0 .+ P' * lam
                qlam = q0 .+ Q' * lam
                gvec = P * uxinv1 .+ Q * xlinv1
                dpsidx = plam ./ ux2 .- qlam ./ xl2
                rex = dpsidx .- xsi .+ eta
                rey = c .+ d .* y .- mu .- lam
                rez = a0 .- zet .- a_mma' * lam
                relam = gvec .- a_mma * z .- y .+ s .- bm
                rexsi = xsi .* (x .- alfa) .- epsvecn
                reeta = eta .* (beta .- x) .- epsvecn
                remu = mu .* y .- epsvecm
                rezet = zet * z .- epsi
                res = lam .* s .- epsvecm

                residu1 = vcat(rex, rey, rez)
                residu2 = vcat(relam, rexsi, reeta, remu, rezet, res)
                residu = vcat(residu1, residu2)
                resinew = norm(residu)
                steg /= 2
            end

            residunorm = resinew
            residumax = norm(residu, Inf)
        end

    epsi *= 0.1
    end
    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
end