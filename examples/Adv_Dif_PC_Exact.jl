using NLsolve

const T0 = 940.0; const Pe = 7.0; const TL = 298.0;
const kₛ = 231.0; const kₗ = 90.7; const cpₛ = 900.0; const cpₗ = 1300.0;
const ρ = 2700.0; const Hf = 396400.0; const Tₗ = 930.0; const Tₛ = 919.0;
const L = 1.0; const bd = 1.0; const modelo = 1;

function alvarez_adv_dif_solidification(
    ks, kl, Cps, Cpl, Tsol, Tliq, rho, To, TL, Pe, Hf, L, modelo, bd
)
    # 1. Propiedades promedio
    Teq = (Tsol + Tliq) / 2.0
    fseq = (Teq - Tliq) / (Tsol - Tliq)
    dfseqdT = 1.0 / (Tsol - Tliq)
    fssol = 1.0
    fsliq = 0.0

    km = fseq * ks + (1.0 - fseq) * kl
    Cpm = fseq * Cps + (1.0 - fseq) * Cpl # Capacidad calorífica aparente (sin Hf)

    alpham = km / (rho * (Cpm - Hf * dfseqdT))
    alphas = ks / (Cps * rho)
    alphal = kl / (Cpl * rho)
    vo = Pe * alphal / L

    # 2. Sistema no lineal EXACTO de tu MATLAB
    function stefan_mushy!(F, S)
        # S[1] = Sl (Liquidus), S[2] = Ss (Solidus)
        
        # f(1)
        F[1] = (Tsol - Tliq) * (Cpm - Hf * dfseqdT) * exp(vo * S[1] / alpham) / (exp(vo * S[2] / alpham) - exp(vo * S[1] / alpham)) - 
               (Tliq - To) * Cpl * exp(vo * S[1] / alphal) / (exp(vo * S[1] / alphal) - 1.0) + 
               Hf * fsliq

        # f(2)
        F[2] = (TL - Tsol) * Cps * exp(vo * S[2] / alphas) / (exp(vo * L / alphas) - exp(vo * S[2] / alphas)) - 
               (Tsol - Tliq) * (Cpm - Hf * dfseqdT) * exp(vo * S[2] / alpham) / (exp(vo * S[2] / alpham) - exp(vo * S[1] / alpham)) + 
               Hf * (1.0 - fssol)
    end

    # Semilla inicial
    S0 = [0.4, 0.6]

    res = nlsolve(stefan_mushy!, S0)

    Sl = res.zero[1]
    Ss = res.zero[2]

    # 3. Perfiles térmicos
    xl = range(0.0, Sl, length=101)
    xm = range(Sl, Ss, length=101)
    xs = range(Ss, L, length=101)

    Tl = @. (Tliq - To) / (exp(vo * Sl / alphal) - 1.0) * (exp(vo * xl / alphal) .- 1.0) + To
    Tm = @. (Tsol - Tliq) / (exp(vo * Ss / alpham) - exp(vo * Sl / alpham)) * (exp(vo * xm / alpham) .- exp(vo * Sl / alpham)) + Tliq
    Ts = @. (TL - Tsol) / (exp(vo * L / alphas) - exp(vo * Ss / alphas)) * (exp(vo * xs / alphas) .- exp(vo * Ss / alphas)) + Tsol

    x = vcat(collect(xl), collect(xm), collect(xs))
    T = vcat(Tl, Tm, Ts)

    return x, T, Ss, Sl
end

x, T, Ss, Sl = alvarez_adv_dif_solidification(kₛ, kₗ, cpₛ, cpₗ, Tₛ, Tₗ, ρ, T0, TL, Pe, Hf, L, modelo, bd)

println("Sl (Liquidus) = ", Sl)
println("Ss (Solidus)  = ", Ss)