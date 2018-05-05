# File created on: 2018-04-26 14:54:00.773449
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com

import bpy
import os
from mathutils import Vector as Vector
from mathutils import Matrix as Matrix

# To view the printed output toggle the system console in "Window -> Toggle System Console"

# This function is called when the DDM operator is selected in Blender.
def DDM_Practical1(context):
	
	os.system('cls')
	
	# TODO: get the triangles for the active mesh and use show_mesh() to display a copy
	print("yeet")
	tris = get_triangles(context)
	show_mesh(tris)

	# TODO: print the euler_characteristic value for the active mesh
	print(euler_characteristic(tris))
	
	# TODO: print whether the active mesh is closed
	
	# TODO: print whether the active mesh is orientable
	
	# TODO: print the genus of the active mesh
	
	# TODO: use show_mesh() to display the maximal_independent_set of the dual of the active object
	
	
	#print(get_triangles(context))

# Builds a mesh using a list of triangles
def show_mesh(triangles):
	# Create a mesh and object first
	mesh = bpy.data.meshes.new("mesh")
	obj = bpy.data.objects.new(bpy.context.scene.objects.active.name + " (copy)", mesh)
	# Link the object to the scene
	scene = bpy.context.scene
	scene.objects.link(obj)
	# Add the vertices of all triangles to a list
	vertices = []
	# Create a list for the faces too
	faces = []
	facecounter = 0
	for t in triangles:
		# Create a face (list of vertices of the triangle) and increment the index of the vertices
		face = []
		# Add all the vertices and face indexes that don't exist yet
		for i in range(0, len(t)):
			# Check if vertex is already added, if yes: add existing vertex to face; If no: create new vertex and add that
			if t[i] in vertices:
				face.append(vertices.index(t[i]))
			else:
				vertices.append(t[i])
				face.append(facecounter)
				facecounter += 1
		# Add the face to the list of faces
		faces.append(face)
	# Add data to the mesh
	mesh.from_pydata(vertices, [], faces)
	# Update mesh changes
	mesh.update()
	
# Returns the faces of the active object as a list of triplets of points
def get_triangles(context):
	# Get the currently active object
	obj = bpy.context.scene.objects.active
	# Create empty triangles list
	triangles = []
	# Get this object's polygons
	polygons = obj.data.polygons
	# For each polygon, split into triangles and add those to the list
	for p in polygons:
		verts = p.vertices
		# Every convex face consists of [vertices - 2] triangles
		for i in range(0, len(verts) - 2):
			tri = []
			tri.append(obj.data.vertices[verts[0]].co)
			tri.append(obj.data.vertices[verts[i + 1]].co)
			tri.append(obj.data.vertices[verts[i + 2]].co)
			triangles.append(tri)
		
	return triangles

def testSquare():
	return [(Vector([0, 0, 0]), Vector([1, 0, 0]), Vector([0, 1, 0])), (Vector([1, 0, 0]), Vector([1, 1, 0]), Vector([0, 1, 0]))]

# Calculates the Euler characteristic of the given list of triangles
def euler_characteristic(triangles):
	# Create a list of all the vertices
	vertices = []
	# Add all vertices to a list without duplicates
	for triangle in triangles:
		for vertex in triangle:
			if vertex not in vertices:
				vertices.append(vertex)
	# Calculate the amount of vertices by getting the length of the vertices list
	v = len(vertices)

	# Create a list of all the edges
	edges = []
	# Sort all the vertex pairs and add all non-duplicates
	for t in triangles:
		x = sortVectorTuple([t[0], t[1]])
		if x not in edges: edges.append(x)
		x = sortVectorTuple([t[0], t[2]])
		if x not in edges: edges.append(x)
		x = sortVectorTuple([t[1], t[2]])
		if x not in edges: edges.append(x)
	# Calculate the amount of edges by getting the length of the edges list
	e = len(edges)

	# Get the amount of faces (the amount of triangles)
	f = len(triangles)

	# Euler characteristic
	return v + f - e

def sortVectorTuple(tuple):
	a = tuple[0]
	b = tuple[1]
	end = (a, b)
	if b.z < a.z: end = (b, a)
	if end[1].y < end[0].y: end = (end[1], end[0])
	if end[1].x < end[0].x: end = (end[1], end[0])
	return end

# Returns whether the given list of triangles is a closed surface
def is_closed(triangles):
	return False
	
# Returns whether the given list of triangles is orientable
def is_orientable(triangles):
	return False
	
# Returns the genus of the given list of triangles
def genus(triangles):
	return -1

# TODO: Fix assignment
def maximal_independent_set(triangles):
	
	# TODO: construct the dual of the mesh
	
	# TODO: find the maximal independent set in this dual
	
	return [(Vector([0, 0, 0]), Vector([1, 0, 0]), Vector([0, 1, 0])), (Vector([1, 0, 0]), Vector([1, 1, 0]), Vector([0, 1, 0]))]
	
