
function plot_field(model, field; title="Resultado FlexiGal", Deformation_Field=nothing, scale=1.0, invert_y=false)
    if !isdefined(Main, :GLMakie)
        @eval Main using GLMakie
    end
    
    # --- Función auxiliar interna para convertir Vector{SVector} a Matrix{Float64} ---
    function _to_matrix(vec_vals)
        N = length(vec_vals)
        N == 0 && return Matrix{Float64}(undef, 0, 0)
        d = length(vec_vals[1])
        M = Matrix{Float64}(undef, N, d)
        @inbounds for i in 1:N, j in 1:d
            M[i, j] = vec_vals[i][j]
        end
        return M
    end

    # 1. Normalización de campos de entrada
    fields = field isa AbstractVector ? field : [field]
    
    # Manejo abreviado: si Deformation_Field = true, usamos el propio 'field'
    def_fields = if Deformation_Field === true
        fields
    elseif isnothing(Deformation_Field)
        nothing
    elseif Deformation_Field isa AbstractVector
        Deformation_Field
    else
        [Deformation_Field]
    end

    # Determinamos la dimensión geométrica d_geo
    coords_1 = get_field_coordinates(fields[1], model)
    d_geo = size(coords_1, 2)
    
    # Contenedores globales
    all_coords = Matrix{Float64}(undef, 0, d_geo)
    all_deformed_coords = Matrix{Float64}(undef, 0, d_geo)
    all_values = []
    
    # 2. Recolección sincronizada por subdominio
    for (idx, f) in enumerate(fields)
        coords_f = get_field_coordinates(f, model)
        vals_f = Get_Point_Values(f)
        
        if !isnothing(def_fields)
            # Asignamos el campo de deformación correspondiente al subdominio
            df = (length(def_fields) >= idx) ? def_fields[idx] : def_fields[1]
            u_f = Get_Point_Values(df)
            u_mat = _to_matrix(u_f)
            
            @assert size(coords_f, 1) == size(u_mat, 1) "Mismatched dimensions: Subdomain $idx coordinates ($(size(coords_f, 1))) do not match deformation field points ($(size(u_mat, 1)))."
            
            def_coords_f = coords_f .+ scale .* u_mat
            all_deformed_coords = vcat(all_deformed_coords, def_coords_f)
        end

        all_coords = vcat(all_coords, coords_f)
        append!(all_values, vals_f)
    end

    # Selección de los puntos finales a graficar
    points_to_plot = isnothing(def_fields) ? all_coords : all_deformed_coords

    # 3. Clasificación de datos
    sample_val = all_values[1]
    T = typeof(sample_val)
    plot_list = []
    
    if T <: Real
        push!(plot_list, (Float64.(all_values), "Escalar"))
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

    # 4. Renderizado con GLMakie
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