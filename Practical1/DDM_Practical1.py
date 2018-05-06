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
from collections import defaultdict
from itertools import combinations
from mathutils import Vector as Vector
from mathutils import Matrix as Matrix

# To view the printed output toggle the system console in "Window -> Toggle System Console"

# This function is called when the DDM operator is selected in Blender.
def DDM_Practical1(context):
	
	os.system('cls')
	
	# TODO: get the triangles for the active mesh and use show_mesh() to display a copy
	tris = get_triangles(context)
	show_mesh(tris)

	# TODO: print the euler_characteristic value for the active mesh
	print("Euler characteristic: ", euler_characteristic(tris))
	
	# TODO: print whether the active mesh is closed
	print("Is closed: ", is_closed(tris))

	# TODO: print whether the active mesh is orientable
	print("Is orientable: ", is_orientable(tris))
	
	# TODO: print the genus of the active mesh
	print("Genus: ", genus(tris))

	# TODO: use show_mesh() to display the maximal_independent_set of the dual of the active object
	#maximal_independent_set(tris)
	
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
	edges = findEdges(triangles)
	# Calculate the amount of edges by getting the length of the edges list
	e = len(edges)

	# Get the amount of faces (the amount of triangles)
	f = len(triangles)

	# Euler characteristic
	return v + f - e

def findEdges(triangles):
	edges = []
	# Sort all the vertex pairs and add all non-duplicates
	for t in triangles:
		x = sortVectorTuple([t[0], t[1]])
		if x not in edges: edges.append(x)
		x = sortVectorTuple([t[0], t[2]])
		if x not in edges: edges.append(x)
		x = sortVectorTuple([t[1], t[2]])
		if x not in edges: edges.append(x)
	return edges

def sortVectorTuple(tuple):
	# Sort tuple with priority X -> Y -> Z
	a = tuple[0]
	b = tuple[1]
	end = (a, b)
	if b.z < a.z: end = (b, a)
	if end[1].y < end[0].y: end = (end[1], end[0])
	if end[1].x < end[0].x: end = (end[1], end[0])
	return end

# Returns whether the given list of triangles is a closed surface
def is_closed(triangles):
	# Get all the edges non-duplicated
	edges = findEdges(triangles)
	# Check for all edges if they're shared by at least two triangles
	for e in edges:
		counter = 0
		for t in triangles:
			if t[0] == e[0] and t[1] == e[1]: counter += 1
			elif t[0] == e[0] and t[2] == e[1]: counter += 1
			elif t[1] == e[0] and t[2] == e[1]: counter += 1
			elif t[1] == e[0] and t[0] == e[1]: counter += 1
			elif t[2] == e[0] and t[0] == e[1]: counter += 1
			elif t[2] == e[0] and t[1] == e[1]: counter += 1
		if counter < 2: return False
	return True
	
# Returns whether the given list of triangles is orientable
def is_orientable(triangles):
	totalNormals = Vector((0, 0, 0))
	for t in triangles:
		dif1 = t[0] - t[2]
		dif2 = t[1] - t[2]
		totalNormals += dif1.cross(dif2)
	print("YEET ", totalNormals)
	return totalNormals != Vector((0, 0, 0))

	
# Returns the genus of the given list of triangles
def genus(triangles):
	# Mesh is not a closed orientable surface, return -1
	if not is_closed(triangles) and not is_orientable(triangles):
		return -1
	else:
		euler = euler_characteristic(triangles)
		# Get the genus with the euler chracteristic formula: χ = 2 − 2g 
		g = (-euler + 2) / 2
		return g

# TODO: Fix assignment
def maximal_independent_set(triangles):
	
	# TODO: construct the dual of the mesh
	# Create vertices in the middle of each face
	faceVerts = []
	for t in triangles:
		x, y, z = .0, .0, .0
		for vert in t:
			x += vert.x
			y += vert.y
			z += vert.z
		cs = len(t)
		x /= cs
		y /= cs
		z /= cs
		faceVerts.append(Vector((x, y, z)))
	# Print for debug
	for t, v in zip(triangles, faceVerts):
		print(t, ': ', v, '\n')
	
	sharededges = defaultdict(list)
	neighbours = defaultdict(list)
	edges = []
	# For each edge, find all the faces that use it
	for i, t in enumerate(triangles):
		for edge in zip(t, t[1:] + t[0]):
			sharededges[sortVectorTuple(edge)].append(i)
	# For each face, find all its neighbours
	for notimportant, ts in sharededges.iteritems():
		for a, b in combinations(ts, 2):
			neighbours[a].append(b)
			neighbours[b].append(a)
	# Calculate all the edges for the dual
	for t, nbs in neighbours.iteritems():
		for nb in nbs:
			edges.append((t, nb))
	
	### faceVerts + edges is the dual graph of the mesh
	
	
	# TODO: find the maximal independent set in this dual
	
	return [(Vector([0, 0, 0]), Vector([1, 0, 0]), Vector([0, 1, 0])), (Vector([1, 0, 0]), Vector([1, 1, 0]), Vector([0, 1, 0]))]
