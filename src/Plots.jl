function plot_field(model, field; title="Resultado FlexiGal", Deformation_Field=nothing, scale=1.0, invert_y=false)
    if !isdefined(Main, :GLMakie)
        @eval Main using GLMakie
    end
    measure = extract_EFGFunction(field).Measure
    gs = measure.gs
    values = Get_Point_Values(field)
    T = eltype(values)
    points_to_plot = if isnothing(Deformation_Field)
        gs
    else
        u_vals = Get_Point_Values(Deformation_Field)
        pos_deformada = [gs[i, 1:2] + scale * u_vals[i] for i in 1:length(u_vals)]
        hcat(first.(pos_deformada), last.(pos_deformada))
    end
    plot_list = []
    if T <: Real
        push!(plot_list, (values, "Scalar"))
    elseif T <: AbstractVector
        D = length(values[1])
        for i in 1:D; push!(plot_list, (getindex.(values, i), "Comp. $i")); end
        push!(plot_list, (norm.(values), "Norm"));
    elseif T <: AbstractMatrix
        R, C = size(values[1])
        for j in 1:C, i in 1:R
            push!(plot_list, (getindex.(values, i, j), "Comp.[$i,$j]"))
        end
        push!(plot_list, (norm.(values), "Norm"))
    end
    def_plot = () -> begin
        n = length(plot_list)
        cols = (n > 1) ? 2 : 1
        rows = ceil(Int, n / cols)
        fig = Main.Figure(size = (1200, 250 * rows)) 
        for (idx, (data, label)) in enumerate(plot_list)
            r = ceil(Int, idx / cols)
            c = mod1(idx, cols)
            gl = fig[r, c] = Main.GridLayout()
            ax = Main.Axis(gl[1, 1], 
                           title = label,
                           aspect = Main.DataAspect(),
                           xlabel = "x", ylabel = "y",
                           yreversed = invert_y) 
            plt = Main.scatter!(ax, points_to_plot[:, 1], points_to_plot[:, 2]; 
                                color = data, 
                                markersize = 4, 
                                colormap = :jet)  
            Main.Colorbar(gl[1, 2], plt, width = 15)
        end 
        Main.Label(fig[0, :], title, fontsize = 20, font = :bold)
        Main.colgap!(fig.layout, 20)
        Main.rowgap!(fig.layout, 20) 
        return fig
    end
    return Base.invokelatest(def_plot)
end