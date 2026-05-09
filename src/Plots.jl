function plot_field(model, field; title="Resultado FlexiGal", Deformation_Field=nothing, scale=1.0, invert_y=false)
    if !isdefined(Main, :GLMakie)
        @eval Main using GLMakie
    end
    
    # 1. Normalización de entrada
    fields = field isa AbstractVector ? field : [field]
    
    # Inicializamos con el tipo del primer elemento para no perder la inferencia
    f1 = extract_FlexiFunction(fields[1])
    d_geo = size(f1.Measure.gs, 2) - 2
    
    # Contenedores con tipos explícitos
    all_coords = Matrix{Float64}(undef, 0, d_geo)
    all_values = [] # Lo dejaremos como Any pero cambiaremos cómo detectamos T
    
    # 2. Recolección
    for f in fields
        fun = extract_FlexiFunction(f)
        gs = fun.Measure.gs 
        coords_f = gs[:, 1:d_geo]
        
        all_coords = vcat(all_coords, coords_f)
        append!(all_values, Get_Point_Values(f))
    end

    # --- EL ARREGLO CLAVE ---
    # Detectamos el tipo del contenido real, no del contenedor
    sample_val = all_values[1]
    T = typeof(sample_val)
    
    # 3. Deformación
    points_to_plot = if isnothing(Deformation_Field)
        all_coords
    else
        def_fields = Deformation_Field isa AbstractVector ? Deformation_Field : [Deformation_Field]
        all_u = []
        for df in def_fields
            append!(all_u, Get_Point_Values(df))
        end
        # Convertimos a matriz estándar para evitar problemas de Adjoint en Makie
        pos = [all_coords[i, :] + scale * all_u[i] for i in 1:length(all_u)]
        reduce(hcat, pos)' |> collect 
    end

    # 4. Clasificación de datos (Igual, pero usando el T corregido)
    plot_list = []
    if T <: Real
        push!(plot_list, (Float64.(all_values), "Escalar")) # Forzamos Float64 por si acaso
    elseif T <: AbstractVector
        D = length(sample_val)
        for i in 1:D; push!(plot_list, (getindex.(all_values, i), "Comp. $i")); end
        push!(plot_list, (norm.(all_values), "Norma"))
    elseif T <: AbstractMatrix
        R, C = size(sample_val)
        for j in 1:C, i in 1:R
            push!(plot_list, (getindex.(all_values, i, j), "Comp.[$i,$j]"))
        end
        push!(plot_list, (norm.(all_values), "Norma"))
    end

    # 5. Graficación
    def_plot = () -> begin
        if isempty(plot_list)
            error("No se encontraron datos para graficar. Revisa el tipo de field.")
        end
        
        n = length(plot_list)
        cols = (n > 1) ? 2 : 1
        rows = ceil(Int, n / cols)
        fig = Main.Figure(size = (n == 1 ? (800, 600) : (1200, 500 * rows))) 

        for (idx, (data, label)) in enumerate(plot_list)
            r = ceil(Int, idx / cols); c = mod1(idx, cols)
            
            # Decidimos si usar Axis o Axis3
            if d_geo == 3
                ax = Main.Axis3(n == 1 ? fig[1, 1] : fig[r, c][1, 1], title = label, aspect = :data)
                plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2], points_to_plot[:, 3]; 
                                    color = data, markersize = 6, colormap = :jet)
                Main.Colorbar(n == 1 ? fig[1, 2] : fig[r, c][1, 2], plt, width = 15)
            else
                ax = Main.Axis(n == 1 ? fig[1, 1] : fig[r, c][1, 1], title = label, 
                               aspect = Main.DataAspect(), yreversed = invert_y)
                plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2]; 
                                    color = data, markersize = 6, colormap = :jet)
                Main.Colorbar(n == 1 ? fig[1, 2] : fig[r, c][1, 2], plt, width = 15)
            end
        end 
        Main.Label(fig[0, :], title, fontsize = 24, font = :bold)
        return fig
    end

    return Base.invokelatest(def_plot)
end