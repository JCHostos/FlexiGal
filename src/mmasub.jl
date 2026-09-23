function mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a_mma, c, d)
    epsimin = 1.0e-7 # Convergence criterion within an outer iteration of subsolv
    
    # Factor for Taylor linearisation of objectiv function and constraints
    raa0 = 1.0e-5 # Initial value for the asymptote factor
    move = 0.5 # Move factor for the asymptotes
    albefa = 0.1 # Alpha-beta factor for the asymptotes
    asyinit = 0.5  # Initial asymptote factor
    asyincr = 1.2 # Asymptote increase factor
    asydecr = 0.7 # Asymptote decrease factor

    eeen = ones(n)
    eeem = ones(m)

    # Calculation of the asymptotes low and upp
    if iter < 2.5
        low = xval .- asyinit .* (xmax .- xmin) #Starting low asymptote
        upp = xval .+ asyinit .* (xmax .- xmin) #Starting upp asymptote
    else
        zzz = (xval .- xold1) .* (xold1 .- xold2) # Fraction to define factor gamma
        factor = ones(n) # Initialize factor for asymptotes (it does not change for xval==xhold1)
        factor[zzz.>0.0] .= asyincr # Assign increase factor for positive fraction
        factor[zzz.<0.0] .= asydecr # Assign decrease factor for negative fraction

        low = xval .- factor .* (xold1 .- low) # Update low asymptote
        upp = xval .+ factor .* (upp .- xold1) # Update upp asymptote

        dx = xmax .- xmin
        lowmax = xval .- 0.01 .* dx # Maximum low asymptote
        uppmin = xval .+ 0.01 .* dx # Minimum upp asymptote
        lowmin = xval .- 10.0 .* dx # Minimum low asymptote
        uppmax = xval .+ 10.0 .* dx # Maximum upp asymptote

        low = max.(low, lowmin) # Correcting low asymptote
        low = min.(low, lowmax)
        upp = min.(upp,uppmax)
        upp = max.(upp,uppmin) # Correcting upp asymptote
    end
    # Calculation of alfa and beta
    zzz1 = low .+ albefa .* (xval .- low) 
    zzz2 = xval .- move .* (xmax .- xmin)
    zzz = max.(zzz1, zzz2)
    alfa = max.(zzz, xmin) # Alpha is merely the maximum among zzz1, zzz2 and xmin

    zzz1 = upp .- albefa .* (upp .- xval)
    zzz2 = xval .+ move .* (xmax .- xmin)
    zzz = min.(zzz1, zzz2)
    beta = min.(zzz, xmax) # Alpha is merely the maximum among zzz1, zzz2 and xmin

    # Calculations of p0, q0, P, Q and b
    xmami = xmax .- xmin #Denominator for approximation of objective function and constraints
    xmamieps = 0.00001 .* eeen # Minimum admissible value for xmami for stability
    xmami = max.(xmami, xmamieps) # Ensure xmami is not too small
    xmamiinv = 1.0 ./ xmami # Putting the denominator as such

    ux1 = upp .- xval #Numerator for p
    ux2 = ux1 .^ 2 #Square numerator for p
    xl1 = xval .- low #Numerator for q
    xl2 = xl1 .^ 2 #Square numerator for q
    uxinv = 1.0 ./ ux1 # This inverse would be for r
    xlinv = 1.0 ./ xl1 # This inverse would be for r
    
    # Linearised objective function
    p0 = max.(df0dx, 0.0)
    q0 = max.(-df0dx, 0.0)

    pq0 = 0.001 .* (p0 .+ q0) .+ raa0 .* xmamiinv # A common term to assemble p and q for linearised objective function
    p0 .+= pq0
    q0 .+= pq0 
    p0 .= p0 .* ux2 # Final form of p for linearised objective function
    q0 .= q0 .* xl2 # Final form of q for linearised objective function
    dfdx = m == 1 ? [dfdx] : dfdx
    # Linearised constraints
    P=zeros(m,n)
    Q=zeros(m,n)
    for i in 1:m
    P[i,:] = max.(dfdx[i]', 0.0)
    Q[i,:] = max.(-dfdx[i]', 0.0)
    end
    PQ = 0.001 .* (P .+ Q) .+ raa0 .* (eeem * xmamiinv') # A common term to assemble P and Q for linearised constraints
    P .+= PQ
    Q .+= PQ
    P .= P .* ux2' # Final form of P for linearised constraints
    Q .= Q .* xl2' # Final form of Q for linearised constraints
    bm = P * uxinv .+ Q * xlinv .- fval #This is -r for constraints
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s =
        subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a_mma, bm, c, d)
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
end