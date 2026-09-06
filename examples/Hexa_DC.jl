using Gmsh

function generar_malla_100p_hexaedros()
    Lx, Ly, Lz = 1.73/2, 0.66/2, 2.5
    z_interfase = 0.75

    if gmsh.isInitialized() == 0
        gmsh.initialize()
    end
    gmsh.model.add("Hexa_Puro_FlexiGal")

    occ = gmsh.model.occ
    b1 = occ.addBox(0.0, 0.0, 0.0,          Lx, Ly, z_interfase)
    b2 = occ.addBox(0.0, 0.0, z_interfase,  Lx, Ly, Lz - z_interfase)

    occ.fragment([(3, b1)], [(3, b2)])
    occ.synchronize()

    vols = gmsh.model.getEntities(3)
    idx_inf = findfirst(v -> gmsh.model.getBoundingBox(3, v[2])[3] < 0.1, vols)
    idx_sup = findfirst(v -> gmsh.model.getBoundingBox(3, v[2])[3] >= 0.1, vols)

    v_inf = vols[idx_inf][2]
    v_sup = vols[idx_sup][2]

    # Configurar divisiones para garantizar HEX8 en ambos volúmenes
    # Misma grilla en X e Y para que coincidan perfectamente en la interfaz
    for v in (v_inf, v_sup)
        surfs = gmsh.model.getBoundary([(3, v)], false)
        curves = unique([c[2] for c in gmsh.model.getBoundary(surfs, false)])

        for c in curves
            bbox = gmsh.model.getBoundingBox(1, c)
            dx = abs(bbox[4] - bbox[1])
            dy = abs(bbox[5] - bbox[2])
            dz = abs(bbox[6] - bbox[3])

            if dx > 0.1; gmsh.model.mesh.setTransfiniteCurve(c, 51); end # 50 divisiones X
            if dy > 0.1; gmsh.model.mesh.setTransfiniteCurve(c, 29); end # 28 divisiones Y
            
            # Divisiones en Z según el bloque
            if dz > 0.1
                if v == v_inf
                    gmsh.model.mesh.setTransfiniteCurve(c, 22) # Densidad alta en MK (21 div)
                else
                    # En la parte superior podemos relajar Z con menos elementos
                    # o usar una progresión para agrandar celdas hacia arriba
                    gmsh.model.mesh.setTransfiniteCurve(c, 21) # Ajustar divisiones Z arriba
                end
            end
        end

        for s in surfs
            gmsh.model.mesh.setTransfiniteSurface(s[2])
        end
        gmsh.model.mesh.setTransfiniteVolume(v)
    end

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(3)

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Extraer Nodos
    _, coords, _ = gmsh.model.mesh.getNodes()
    xyz = reshape(coords, 3, :)' |> Matrix{Float64}

    # Limpieza de tolerancia
    xyz[:, 1] .= clamp.(xyz[:, 1], 0.0, Lx)
    xyz[:, 2] .= clamp.(xyz[:, 2], 0.0, Ly)
    xyz[:, 3] .= clamp.(xyz[:, 3], 0.0, Lz)

    # Extraer Conectividad
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(3)
    idx_hex8 = findfirst(==(5), elemTypes)

    if idx_hex8 !== nothing
        conn = reshape(elemNodeTags[idx_hex8], 8, :)' |> Matrix{Int}
        println("¡Malla 100% Hexaédrica generada!")
        println("Total Nodos: ", size(xyz, 1))
        println("Total HEX8: ", size(conn, 1))
        
        gmsh.finalize()
        return xyz, conn
    else
        gmsh.finalize()
        error("No se generaron hexaedros.")
    end
end

xyz, conn = generar_malla_100p_hexaedros()