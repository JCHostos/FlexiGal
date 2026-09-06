using Gmsh
using LinearAlgebra

# 1. Geometría del Dominio
Lx, Ly = 1.0, 1.0         # Cuadrado [-0.5, 0.5] x [-0.5, 0.5]
r_circ = 0.25             # Radio del orificio central

# Tamaño de elemento global refinado (ajustar h según la densidad deseada)
h_refinado = 0.005

# 2. Inicializar Gmsh
gmsh.initialize()
gmsh.model.add("SquareCav_Refined_Uniform")

occ = gmsh.model.occ

# Crear el rectángulo base centrado en (0,0)
rect = occ.addRectangle(-Lx/2, -Ly/2, 0.0, Lx, Ly)

# Crear el disco central (r = 0.25)
disco = occ.addDisk(0.0, 0.0, 0.0, r_circ, r_circ)

# Restar el disco del rectángulo
occ.cut([(2, rect)], [(2, disco)])
occ.synchronize()

# 3. Tamaño de elemento uniforme en TODO el dominio
gmsh.option.setNumber("Mesh.MeshSizeMin", h_refinado)
gmsh.option.setNumber("Mesh.MeshSizeMax", h_refinado)

# Algoritmo Frontal-Delaunay para alta calidad isotrópica
gmsh.option.setNumber("Mesh.Algorithm", 6)

# 4. Generar y optimizar malla 2D
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")

# REORDENAMIENTO CRÍTICO: Garantiza correlación 1:1 entre filas de 'xy' e índices de 'conn'
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()

# 5. Extraer Nodos (xy - Matriz N x 2)
_, coords, _ = gmsh.model.mesh.getNodes()
xy = reshape(coords, 3, :)'[1:end, 1:2] |> Matrix{Float64}

# 6. Extraer Conectividad (conn - Matriz M x 3)
elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
idx_tri3 = findfirst(==(2), elemTypes)
conn = reshape(elemNodeTags[idx_tri3], 3, :)' |> Matrix{Int}

gmsh.finalize()