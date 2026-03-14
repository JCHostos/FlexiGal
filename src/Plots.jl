function plot_field(model, field; title="Resultado FlexiGal", Deformation_Field=nothing, scale=1.0, invert_y=false)
    if !isdefined(Main, :GLMakie)
        @eval Main using GLMakie
    end
    
    efg_f = extract_EFGFunction(field)
    measure = efg_f.Measure
    gs = measure.gs 
    
    # La dimensión física real (restando Jacobiano y Peso)
    dim_geo = size(gs, 2) - 2 
    
    values = Get_Point_Values(field)
    T = eltype(values)
    
    # Coordenadas según dimensión real
    coords_puras = gs[:, 1:dim_geo]
    
    # Manejo de deformación
    points_to_plot = if isnothing(Deformation_Field)
        coords_puras
    else
        u_vals = Get_Point_Values(Deformation_Field)
        pos_deformada = [coords_puras[i, :] + scale * u_vals[i] for i in 1:length(u_vals)]
        reduce(hcat, pos_deformada)'
    end

    # Lista de datos a graficar
    plot_list = []
    if T <: Real
        push!(plot_list, (values, "Escalar"))
    elseif T <: AbstractVector
        D = length(values[1])
        for i in 1:D; push!(plot_list, (getindex.(values, i), "Comp. $i")); end
        push!(plot_list, (norm.(values), "Norma"))
    elseif T <: AbstractMatrix
        R, C = size(values[1])
        for j in 1:C, i in 1:R
            push!(plot_list, (getindex.(values, i, j), "Comp.[$i,$j]"))
        end
        push!(plot_list, (norm.(values), "Norma"))
    end

    def_plot = () -> begin
        n = length(plot_list)
        cols = (n > 1) ? 2 : 1
        rows = ceil(Int, n / cols)
        
        # Tamaño de ventana generoso
        fig = Main.Figure(size = (n == 1 ? (1000, 800) : (1200, 500 * rows))) 

        for (idx, (data, label)) in enumerate(plot_list)
            r = ceil(Int, idx / cols)
            c = mod1(idx, cols)
            
            if n == 1
                # CASO ÚNICO CAMPO: Bypasseamos el GridLayout para maximizar el área
                if dim_geo == 3
                    ax = Main.Axis3(fig[1, 1], title = label, aspect = :data,
                                    xlabel = "x", ylabel = "y", zlabel = "z")
                    plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2], points_to_plot[:, 3]; 
                                        color = data, markersize = 6, colormap = :jet)
                else
                    ax = Main.Axis(fig[1, 1], title = label, aspect = Main.DataAspect(),
                                   xlabel = "x", ylabel = "y", yreversed = invert_y)
                    plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2]; 
                                        color = data, markersize = 6, colormap = :jet)
                end
                
                # Barra de colores directamente en la segunda columna de la figura
                Main.Colorbar(fig[1, 2], plt, width = 15)
                
            else
                # CASO MÚLTIPLES CAMPOS: Usamos GridLayout para organizar
                gl = fig[r, c] = Main.GridLayout()
                if dim_geo == 3
                    ax = Main.Axis3(gl[1, 1], title = label, aspect = :data)
                    plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2], points_to_plot[:, 3]; 
                                        color = data, markersize = 4, colormap = :jet)
                else
                    ax = Main.Axis(gl[1, 1], title = label, aspect = Main.DataAspect(),
                                   xlabel = "x", ylabel = "y", yreversed = invert_y)
                    plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2]; 
                                        color = data, markersize = 4, colormap = :jet)
                    Main.colsize!(gl, 1, Main.Relative(0.85)) # Ajuste solo para múltiples
                end
                Main.Colorbar(gl[1, 2], plt, width = 12)
            end
        end 

        Main.Label(fig[0, :], title, fontsize = 24, font = :bold)
        return fig
    end

    return Base.invokelatest(def_plot)
end