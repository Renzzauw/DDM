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
	
	# Get the triangles for the active mesh and use show_mesh() to display a copy
	tris = get_triangles(context)
	show_mesh(tris)

	# Print the euler_characteristic value for the active mesh
	print("Euler characteristic: ", euler_characteristic(tris))
	
	# Print whether the active mesh is closed
	print("Is closed: ", is_closed(tris))

	# Print whether the active mesh is orientable
	print("Is orientable: ", is_orientable(tris))
	
	# Print the genus of the active mesh
	print("Genus: ", genus(tris))

	# Use show_mesh() to display the maximal_independent_set of the dual of the active object
	show_mesh(maximal_independent_set(tris))

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
	return totalNormals != Vector((0, 0, 0))

	
# Returns the genus of the given list of triangles
def genus(triangles):
	# Mesh is not a closed and an orientable surface, return -1
	if not (is_closed(triangles) and is_orientable(triangles)):
		return -1
	else:
		euler = euler_characteristic(triangles)
		# Get the genus with the euler chracteristic formula: χ = 2 − 2g 
		g = (-euler + 2) / 2
		return g

# Returns the maximal independent set for a given list of triangles
def maximal_independent_set(triangles):
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
	
	# Create dictionaries for shared edges and neighbouring triangles
	sharededges = defaultdict(list)
	neighbours = defaultdict(list)
	edges = set()
	# For each edge, find all the faces that use it
	for i, t in enumerate(triangles):
		for edge in zip(t, t[1:] + t[:1]):
			eList = []
			for v in edge:
				eList.append(vec2tup(v))
			sharededges[tuple(sorted(eList))].append(i)
	# For each face, find all its neighbours
	for notimportant, ts in sharededges.items():
		for a, b in combinations(ts, 2):
			neighbours[a].append(b)
			neighbours[b].append(a)
	# Calculate all the edges for the dual
	for t, nbs in neighbours.items():
		for nb in nbs:
			edges.add(tuple(sorted(list((t, nb)))))
	edges = sorted(list(edges))
	edgeCoords = []
	for edge in edges:
		edgeCoords.append((faceVerts[edge[0]], faceVerts[edge[1]]))
	# faceVerts + edgeCoords is now the dual graph of the mesh
	
	# Create lists for the maximal independent set and the range of that set
	max_ind_set = []
	rangeOfSet = []
	# For each dual vertex, test if it's in the range of the current max ind set, if not: add it and recalculate the range
	for v in faceVerts:
		if vec2tup(v) not in rangeOfSet:
			max_ind_set.append(v)
			rangeOfSet = calcRangeOfSet(max_ind_set, edgeCoords)

	# Link the original triangles to the maximal independent set
	outputTris = []
	for v in max_ind_set:
		outputTris.append(triangles[faceVerts.index(v)])

	return outputTris

# Calculate the range of 'inSet' using 'edges'
def calcRangeOfSet(inSet, edges):
	outSet = set()
	for v in inSet:
		# Add each starting vertex to the range
		outSet.add(vec2tup(v))
		for e in edges:
			# If the current starting vertex is in an edge, add the other part of the edge to the range
			if e[0] == v: outSet.add(vec2tup(e[1]))
			if e[1] == v: outSet.add(vec2tup(e[0]))
	return list(outSet)

# Function to return the Vector as a tuple
def vec2tup(v):
	return (v.x, v.y, v.z)