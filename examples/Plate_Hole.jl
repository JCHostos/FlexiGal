using Gmsh
using LinearAlgebra

function generar_malla_placa_orificio(; R=1.0, L=10.0, n_r=88, n_t=48, prog=1.02)
    gmsh.initialize()
    gmsh.model.add("Placa_Orificio_2D")

    local xy::Matrix{Float64}
    local conn::Matrix{Int}

    try
        geo = gmsh.model.geo

        # Punto auxiliar (centro de los arcos)
        p_center = geo.addPoint(0.0, 0.0, 0.0)

        # Puntos sobre la circunferencia (R)
        p1 = geo.addPoint(R, 0.0, 0.0)
        p2 = geo.addPoint(R * cos(pi / 4), R * sin(pi / 4), 0.0)
        p3 = geo.addPoint(0.0, R, 0.0)

        # Puntos del contorno exterior
        p4 = geo.addPoint(L, 0.0, 0.0)
        p5 = geo.addPoint(L, L, 0.0)
        p6 = geo.addPoint(0.0, L, 0.0)

        # Arcos
        c_arc1 = geo.addCircleArc(p1, p_center, p2)
        c_arc2 = geo.addCircleArc(p2, p_center, p3)

        # Bordes rectos
        c_bot  = geo.addLine(p1, p4)
        c_rgt  = geo.addLine(p4, p5)
        c_diag = geo.addLine(p2, p5)
        c_top  = geo.addLine(p5, p6)
        c_lft  = geo.addLine(p3, p6)

        cl1 = geo.addCurveLoop([c_bot, c_rgt, -c_diag, -c_arc1])
        s1 = geo.addPlaneSurface([cl1])

        cl2 = geo.addCurveLoop([c_diag, c_top, -c_lft, -c_arc2])
        s2 = geo.addPlaneSurface([cl2])

        geo.synchronize()

        # Malla estructurada transfinita
        gmsh.model.mesh.setTransfiniteCurve(c_arc1, n_t)
        gmsh.model.mesh.setTransfiniteCurve(c_arc2, n_t)
        gmsh.model.mesh.setTransfiniteCurve(c_rgt, n_t)
        gmsh.model.mesh.setTransfiniteCurve(c_top, n_t)

        gmsh.model.mesh.setTransfiniteCurve(c_bot, n_r, "Progression", prog)
        gmsh.model.mesh.setTransfiniteCurve(c_diag, n_r, "Progression", prog)
        gmsh.model.mesh.setTransfiniteCurve(c_lft, n_r, "Progression", prog)

        gmsh.model.mesh.setTransfiniteSurface(s1, "Left", [p1, p4, p5, p2])
        gmsh.model.mesh.setRecombine(2, s1)

        gmsh.model.mesh.setTransfiniteSurface(s2, "Left", [p2, p5, p6, p3])
        gmsh.model.mesh.setRecombine(2, s2)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()

        # Extraer datos de nodos y elementos
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = reshape(coords, 3, :)' |> Matrix{Float64}

        elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
        idx_quad4 = findfirst(==(3), elemTypes)
        idx_quad4 === nothing && error("No se encontraron elementos QUAD4 en la malla.")

        conn_gmsh = reshape(elemNodeTags[idx_quad4], 4, :)' |> Matrix{Int}

        # Renumeración compacta para eliminar nodos huérfanos o no usados
        used_tags = sort(unique(vec(conn_gmsh)))
        tag_to_row = Dict(Int(tag) => i for (i, tag) in enumerate(nodeTags))

        xy = Matrix{Float64}(undef, length(used_tags), 2)
        for (i, tag) in enumerate(used_tags)
            row = tag_to_row[Int(tag)]
            xy[i, 1] = xyz[row, 1]
            xy[i, 2] = xyz[row, 2]
        end

        tag_to_compact = Dict(tag => i for (i, tag) in enumerate(used_tags))
        conn = Matrix{Int}(undef, size(conn_gmsh))
        for i in axes(conn_gmsh, 1)
            for j in axes(conn_gmsh, 2)
                tag = conn_gmsh[i, j]
                conn[i, j] = tag_to_compact[tag]
            end
        end

    finally
        gmsh.finalize()
    end

    return xy, conn
end

# Ejemplo de uso:
xy, conn = generar_malla_placa_orificio()