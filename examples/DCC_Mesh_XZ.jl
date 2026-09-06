using Gmsh

function generar_malla_2d_xz()
    Lx, Lz = 1.73 / 2, 2.5
    z_interfase = 0.75

    if gmsh.isInitialized() == 0
        gmsh.initialize()
    end
    gmsh.model.add("Quad_2D_XZ")

    occ = gmsh.model.occ

    # =========================================================================
    # 1. GEOMETRÍA EN EL PLANO XZ (y = 0)
    # =========================================================================
    p1 = occ.addPoint(0.0, 0.0, 0.0)
    p2 = occ.addPoint(Lx,  0.0, 0.0)
    p3 = occ.addPoint(Lx,  0.0, z_interfase)
    p4 = occ.addPoint(0.0, 0.0, z_interfase)
    p5 = occ.addPoint(Lx,  0.0, Lz)
    p6 = occ.addPoint(0.0, 0.0, Lz)

    # Bloque inferior (z <= 0.75)
    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p1)

    # Bloque superior (z > 0.75)
    l5 = occ.addLine(p3, p5)
    l6 = occ.addLine(p5, p6)
    l7 = occ.addLine(p6, p4)

    cl1 = occ.addCurveLoop([l1, l2, l3, l4])
    s1  = occ.addPlaneSurface([cl1])

    cl2 = occ.addCurveLoop([-l3, l5, l6, l7])
    s2  = occ.addPlaneSurface([cl2])

    occ.synchronize()

    # =========================================================================
    # 2. CONFIGURACIÓN DE LA MALLA 2D
    # =========================================================================
    # Bloque Inferior: Estructurado
    gmsh.model.mesh.setTransfiniteCurve(l1, 51) # 50 elementos en X
    gmsh.model.mesh.setTransfiniteCurve(l3, 51) # 50 elementos en interfase X
    gmsh.model.mesh.setTransfiniteCurve(l2, 22) # 21 elementos en Z
    gmsh.model.mesh.setTransfiniteCurve(l4, 22) # 21 elementos en Z
    gmsh.model.mesh.setTransfiniteSurface(s1)

    # Bloque Superior: No estructurado
    gmsh.model.mesh.setTransfiniteCurve(l6, 17) # 16 elementos arriba en X
    gmsh.model.mesh.setTransfiniteCurve(l5, 17) # 16 elementos en Z
    gmsh.model.mesh.setTransfiniteCurve(l7, 17) # 16 elementos en Z

    # Recombinar AMBAS superficies para forzar QUADS
    gmsh.model.mesh.setRecombine(2, s1) # <--- AQUÍ ESTABA LA FALLA
    gmsh.model.mesh.setRecombine(2, s2)

    # Algoritmos de mallado 2D cuadrilátero
    gmsh.option.setNumber("Mesh.Algorithm", 8)              # Frontal-Delaunay Quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # Blossom

    # =========================================================================
    # 3. GENERACIÓN DE MALLA 2D
    # =========================================================================
    gmsh.model.mesh.generate(2)

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Extraer Nodos
    _, coords, _ = gmsh.model.mesh.getNodes()
    xyz = reshape(coords, 3, :)' |> Matrix{Float64}
    xz = xyz[:, [1, 3]]

    xz[:, 1] .= clamp.(xz[:, 1], 0.0, Lx)
    xz[:, 2] .= clamp.(xz[:, 2], 0.0, Lz)

    # Extraer Conectividad (QUAD4 = Tipo 3 en Gmsh)
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)

    all_quad_tags = Int[]
    for (type, tags) in zip(elemTypes, elemNodeTags)
        if type == 3
            append!(all_quad_tags, tags)
        end
    end

    if !isempty(all_quad_tags)
        conn = reshape(all_quad_tags, 4, :)' |> Matrix{Int}
        println("¡Malla 2D (XZ) de Cuadriláteros corregida!")
        println("Total Nodos 2D: ", size(xz, 1))
        println("Total QUAD4:   ", size(conn, 1))
        
        gmsh.finalize()
        return xz, conn
    else
        gmsh.finalize()
        error("No se generaron cuadriláteros en 2D.")
    end
end

xz, conn = generar_malla_2d_xz()