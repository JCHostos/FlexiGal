using Gmsh
using LinearAlgebra

# 1. Definición del Dominio
Lx, Ly, Lz = 1.73/2, 0.66/2, 2.5

# Tamaños de elemento deseados (Tetraedros)
h_fino   = 0.02  # Tamaño para zona MK (z <= 0.9)
h_grueso = 0.09  # Tamaño para zona FEM (z > 0.9)

# 2. Inicializar Gmsh
gmsh.initialize()
gmsh.model.add("Dominio_MK_FEM")

# Crear la caja 3D en la API OpenCASCADE
occ = gmsh.model.occ
cubo = occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
occ.synchronize()

# --- CONFIGURACIÓN PARA REDUCIR EXCENTRICIDAD ---

# Algoritmo 3D: 4 = Frontal-Delaunay (genera tetraedros muy isotrópicos)
gmsh.option.setNumber("Mesh.Algorithm3D", 4)

# 3. Definir campo de densidad con TRANSICIÓN SUAVE
field_box = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field_box, "VIn",       h_fino)
gmsh.model.mesh.field.setNumber(field_box, "VOut",      h_grueso)
gmsh.model.mesh.field.setNumber(field_box, "XMin",      0.0)
gmsh.model.mesh.field.setNumber(field_box, "XMax",      Lx)
gmsh.model.mesh.field.setNumber(field_box, "YMin",      0.0)
gmsh.model.mesh.field.setNumber(field_box, "YMax",      Ly)
gmsh.model.mesh.field.setNumber(field_box, "ZMin",      0.0)
gmsh.model.mesh.field.setNumber(field_box, "ZMax",      0.75)

# Thickness: crea una rampa de transición de 0.15 m alrededor de los límites de la caja
gmsh.model.mesh.field.setNumber(field_box, "Thickness", 0.1)

# Asignar como campo de fondo para la malla
gmsh.model.mesh.field.setAsBackgroundMesh(field_box)

# 4. Generar malla 3D de tetraedros
gmsh.model.mesh.generate(3)

# OPTIMIZACIÓN: Elimina tetraedros colapsados/excéntricos y maximiza el radio inscripto
gmsh.model.mesh.optimize("Netgen")

# Asegurar que los IDs de los nodos sean 1, 2, ..., N continuos
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()

# 5. Extraer Nodos (xyz)
_, coords, _ = gmsh.model.mesh.getNodes()
xyz = reshape(coords, 3, :)' |> Matrix{Float64} # Matriz N x 3

# 6. Extraer Conectividad (conn) para Tetraedros (Tipo 4 de Gmsh = TET4)
elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(3)
idx_tet4 = findfirst(==(4), elemTypes) # Buscar el tipo TET4
conn = reshape(elemNodeTags[idx_tet4], 4, :)' |> Matrix{Int} # Matriz M x 4

# Finalizar Gmsh
gmsh.finalize()