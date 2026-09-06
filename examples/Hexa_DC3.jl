using Gmsh

function generar_malla_100p_hexaedros()
    Lx, Ly, Lz = 1.73 / 2, 0.66 / 2, 2.5
    z_interfase = 0.75

    if gmsh.isInitialized() == 0
        gmsh.initialize()
    end
    gmsh.model.add("Hexa_Puro_FlexiGal")

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
    l3 = occ.addLine(p3, p4) # Línea de interfase compartida
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
    # Bloque Inferior: Estructurado (50x21 cuadriláteros)
    gmsh.model.mesh.setTransfiniteCurve(l1, 51) # 50 elementos en X
    gmsh.model.mesh.setTransfiniteCurve(l3, 51) # 50 elementos en interfase X
    gmsh.model.mesh.setTransfiniteCurve(l2, 22) # 21 elementos en Z
    gmsh.model.mesh.setTransfiniteCurve(l4, 22) # 21 elementos en Z
    gmsh.model.mesh.setTransfiniteSurface(s1)

    # Bloque Superior: No estructurado (perímetro par: 50+16+16+16 = 98)
    gmsh.model.mesh.setTransfiniteCurve(l6, 17) # 16 elementos en X arriba
    gmsh.model.mesh.setTransfiniteCurve(l5, 17) # 16 elementos en Z
    gmsh.model.mesh.setTransfiniteCurve(l7, 17) # 16 elementos en Z
    gmsh.model.mesh.setRecombine(2, s2)

    # =========================================================================
    # 3. EXTRUSIÓN EN Y (recombine = true directo para HEX8)
    # =========================================================================
    ny = 28 # 28 elementos en Y
    
    occ.extrude([(2, s1)], 0.0, Ly, 0.0, [ny], [1.0], true)
    occ.extrude([(2, s2)], 0.0, Ly, 0.0, [ny], [1.0], true)

    occ.synchronize()

    # Algoritmos de recombinación 2D
    gmsh.option.setNumber("Mesh.Algorithm", 8)              # Frontal-Delaunay Quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # Blossom

    # Generación de la malla 3D
    gmsh.model.mesh.generate(3)

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Extraer Nodos
    _, coords, _ = gmsh.model.mesh.getNodes()
    xyz = reshape(coords, 3, :)' |> Matrix{Float64}

    # Limpieza por tolerancia numérica
    xyz[:, 1] .= clamp.(xyz[:, 1], 0.0, Lx)
    xyz[:, 2] .= clamp.(xyz[:, 2], 0.0, Ly)
    xyz[:, 3] .= clamp.(xyz[:, 3], 0.0, Lz)

    # Extraer Conectividad de todos los bloques HEX8 (Tipo 5)
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(3)

    all_hex8_tags = Int[]
    for (type, tags) in zip(elemTypes, elemNodeTags)
        if type == 5
            append!(all_hex8_tags, tags)
        end
    end

    if !isempty(all_hex8_tags)
        conn = reshape(all_hex8_tags, 8, :)' |> Matrix{Int}
        println("¡Malla 100% Hexaédrica generada exitosamente!")
        println("Total Nodos: ", size(xyz, 1))
        println("Total HEX8:  ", size(conn, 1))
        
        gmsh.finalize()
        return xyz, conn
    else
        gmsh.finalize()
        error("No se generaron hexaedros.")
    end
end

xyz, conn = generar_malla_100p_hexaedros()