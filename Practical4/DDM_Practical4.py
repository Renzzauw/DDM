# File created on: 2018-06-07 13:31:55.107728
#
# CREATED BY:
# Renzo Schindeler 5964962
# Kasper Nooteboom 5845866
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com
# - Do not modify any of the signatures provided.
# - You may add as many functions to the assignment as you see fit but try to avoid global variables.

import ddm
from mathutils import Vector
import bpy

# Place additional imports here
import math
import bmesh

class Mesh():

	# Construct a new mesh according to some data
	def __init__(self, vertices, faces):
	
		# The vertices are stored as a list of vectors
		self.vertices = vertices
		
		# The faces are stored as a list of triplets of vertex indices
		self.faces = faces
	
		# The uv-coordinates are stored as a list of vectors, with their indices corresponding to the indices of the vertices (there is exactly one uv-coordinate per vertex)
		self.uv_coordinates = []
		for vertex in vertices:
			self.uv_coordinates.append(Vector( (vertex[0], vertex[2]) ))
	
		self.build_edge_list()
	
	# This function builds an edge list from the faces of the current mesh and stores it internally.
	# Make sure that each edge is unique. Remember that order does NOT matter, e.g. (1, 0) is the same edge as (0, 1).
	# The indices of the edges should correspond to the indices of the weights in your weight calculation (e.g. both lists have the same size).
	# All subsequent calls that do something with edges should return their indices. Getting the actual edge can then be done by calling get_edge(index).
	def build_edge_list(self):
	
		# Get the faces
		faces = self.get_faces()

		result = []

		for face in faces:
			# Get each vertex of a face
			v1 = face[0]
			v2 = face[1]
			v3 = face[2]
			# Get all the edges of a face (including mirrored edges)
			tuple1 = (v1, v2)
			tuple2 = (v2, v3)
			tuple3 = (v3, v1)
			tuple1mirror = (v2, v1)
			tuple2mirror = (v3, v2)
			tuple3mirror = (v1, v3)
			# Add the edges to the list
			if tuple1 not in result and tuple1mirror not in result:
				result.append(tuple1)
			if tuple2 not in result and tuple2mirror not in result:
				result.append(tuple2)
			if tuple3 not in result and tuple3mirror not in result:
				result.append(tuple3)	

		self.edges = result
	
	# ACCESSORS
	
	def get_vertices(self):
		return self.vertices
		
	def get_vertex(self, index):
		return self.vertices[index]
	
	def get_edges(self):
		return self.edges
		
	def get_edge(self, index):
		return self.edges[index]
	
	def get_faces(self):
		return self.faces
		
	def get_face(self, index):
		return self.faces[index]
		
	def get_uv_coordinates(self):
		return self.uv_coordinates
		
	def get_uv_coordinate(self, index):
		return self.uv_coordinates[index]
	
	# Returns the list of vertex coordinates belonging to a face.
	def get_face_vertices(self, face):
		return [ self.get_vertex(face[0]), self.get_vertex(face[1]), self.get_vertex(face[2]) ]
	
	# Looks up the edges belonging to this face in the edge list and returns their INDICES (not value). Make sure that each edge is unique (recall that (1, 0) == (0, 1)). These should match the order of your weights.
	def get_face_edges(self, face):
			
		# Get the list of all edges in the Mesh
		edges = self.get_edges()
		# Get each edge from the given face (including mirrored identical edges)
		edge1 = (face[0], face[1])
		edge2 = (face[1], face[2])
		edge3 = (face[2], face[0])
		edge1mirror = (face[1], face[0])
		edge2mirror = (face[2], face[1])
		edge3mirror = (face[0], face[2])
		# Create a list for the end result
		result = []
		# Get edge indices
		if edge1 in edges:
			result.append(edges.index(edge1))
		else:
			result.append(edges.index(edge1mirror))
		if edge2 in edges:
			result.append(edges.index(edge2))
		else:
			result.append(edges.index(edge2mirror))
		if edge3 in edges:
			result.append(edges.index(edge3))
		else:
			result.append(edges.index(edge3mirror))
		
		return result
	
	# Returns the vertex coordinates of the vertices of the given edge (a pair of vertex indices e.g. (0,1) ) 
	def get_edge_vertices(self, edge):
		return [ self.get_vertex(edge[0]), self.get_vertex(edge[1])]
	
	# Returns the flap of the given edge belonging to edge_index, that is faces connected to the edge: 1 for a boundary edge, 2 for internal edges
	def get_flaps(self, edge_index):
		
		# Watch out: edges might be on the boundary

		# Get all the faces
		faces = self.get_faces()
		# Get the edge from the given edge index
		edge = self.get_edge(edge_index)
		# Create a list to return
		result = []
		# Get the edges from each face
		for face in faces:
			edge1 = (face[0], face[1])
			edge2 = (face[1], face[2])
			edge3 = (face[2], face[0])
			edge1mirror = (face[1], face[0])
			edge2mirror = (face[2], face[1])
			edge3mirror = (face[0], face[2])
			# Check if the edge exists in the list of faces (check each edge and its identical mirror for each face)
			if edge == edge1 or edge == edge2 or edge == edge3 or edge == edge1mirror or edge == edge2mirror or edge == edge3mirror:
				result.append(face)

		return result
		
	# Returns the length of the given edge with edge_index
	def get_edge_length(self, edge_index):
		
		# Get the edge from the given index
		edge = self.get_edge(edge_index)
		# Get the vertex positions
		v1 = self.get_vertex(edge[0])
		v2 = self.get_vertex(edge[1])
		# Calculate the length of the edge with the edge distances
		xdist = v1[0] - v2[0]
		ydist = v1[1] - v2[1]
		zdist = v1[2] - v2[2]
		# square root (x-distance squared + y-distance squared + z-distance squared)
		result = (xdist**2 + ydist**2 + zdist**2)**(.5)

		return result
		
	# Returns whether the edge has two adjacent faces
	def is_boundary_edge(self, edge_index):
				
		# Get the flaps of the edge at the given edge index
		flaps = self.get_flaps(edge_index)
		# Check the amount of faces:
		# 1 			>> boundary edge found, boundary edge has only 1 face	>> True
		# Anything else >> not a boundary edge									>> False
		return len(flaps) == 1
	
	# Returns the boundary of the mesh by returning the indices of the edges (from the internal edge list) that lie around the boundary.
	def boundary_edges(self):
		
		# Get all the edges
		edges = self.get_edges()

		# Get the flaps with length 1 (aka the boundary edges)
		result = []
		
		for index in range(0, len(edges)):
			#if self.is_boundary_edge(self.get_flaps(index)):
			if self.is_boundary_edge(index):
				result.append(index)
		
		return result
		
	# Place any other accessors you need here
	
# This function is called when the DDM operator is selected in Blender.
def DDM_Practical4(context):
		
	# Construct a Mesh class instance from the active object
	M = get_mesh()

	# show_mesh on a copy of the active mesh with uniform UV coordinates, call this mesh "Uniform"
	show_mesh(Convex_Boundary_Method(M, uniform_weights(M), 1), "Uniform")
	
	# show_mesh on a copy of the active mesh with cot UV coordinates, call this mesh "Cot"
	show_mesh(Convex_Boundary_Method(M, cotan_weights(M), 1), "Cot")

	# TODO: show_mesh on a copy of the active mesh with boundary free UV coordinates, call this mesh "LSCM"
	show_mesh(LSCM(M), "LSCM")

# You may place extra functions here

# Slices a list of triplets and returns two lists of triplets based on a list of fixed columns
# For example if you have a set of triplets T from a matrix with 8 columns, and the fixed columns are [2, 4, 7]
# then all triplets that appear in column [1, 3, 5, 6] are put into "right_triplets" and all triplets that appear in column [2, 4, 7] are put into the set "left_triplets"
def slice_triplets(triplets, fixed_colums):

	left_triplets = []
	right_triplets = []

	# First find the complement of the fixed column set, by finding the maximum column number that appear in the triplets
	max_column = 0
	
	for triplet in triplets:
		if (triplet[1] > max_column):
			max_column = triplet[1]
	
	# and constructing the other columns from those
	other_columns = [x for x in range(0, max_column + 1) if x not in fixed_colums]
	
	# Now split the triplets
	for triplet in triplets:
	
		if (triplet[1] in fixed_colums):
			new_column_index = fixed_colums.index(triplet[1])
			left_triplets.append( (triplet[0], new_column_index, triplet[2]) )
		else:
			new_column_index = other_columns.index(triplet[1])
			right_triplets.append( (triplet[0], new_column_index, triplet[2]) )
			
	return (left_triplets, right_triplets)

# Returns the weights for each edge of mesh M, e.g. a list of real numbers such that the index matches the index of the edge list in M.
def cotan_weights(M):
	
	# Get all edges
	edges = M.get_edges()

	weights = []
	for edge in range(0, len(edges)):
		if M.is_boundary_edge(edge):
			# Skip boundary edges
			continue
		else:
			# Get the flaps
			edgeFlaps = M.get_flaps(edge)
			edges1 = M.get_face_edges(edgeFlaps[0])
			edges2 = M.get_face_edges(edgeFlaps[1])
			# Get the edges of both faces excluding the shared edge
			edges1.remove(edge)
			edges2.remove(edge)
			# Calculate the angles between these edges
			# Angle of face 1
			a = M.get_edge_length(edges1[0])
			b = M.get_edge_length(edges1[1])
			sharedEdge = M.get_edge_length(edge)
			angle1 = math.acos(( a**2 + b**2 - sharedEdge**2) / (2 * a * b))
			# Angle of face 2
			c = M.get_edge_length(edges2[0])
			d = M.get_edge_length(edges2[1])
			sharedEdge = M.get_edge_length(edge)
			angle2 = math.acos(( c**2 + d**2 - sharedEdge**2) / (2 * c * d))
			# Calculate final weight
			weigth = (math.atan(angle1) + math.atan(angle2)) / 2
			weights.append(weigth)

	return weights
	
# Same as above but for uniform weights
def uniform_weights(M):

	# Get all edges
	edges = M.get_edges()
	boundaryEdges = M.boundary_edges()
	# Get the amount of weigths
	weigthsAmount = len(edges) - len(boundaryEdges)
	# Create a list of 1s
	weights = []
	for i in range(0, weigthsAmount):
		weights.append(1)

	return weights
	
# Given a set of weights, return M with the uv-coordinates set according to the passed weights
def Convex_Boundary_Method(M, weights, r):

	# /// 1.1.1 formula (1)

	# Get all the vertices (||V||)
	V = M.get_vertices()
	# Get all the edges
	edges = M.get_edges()
	# Get the boundary edges
	boundaryEdges = M.boundary_edges()
	# Get boundary lengths
	boundaryLengths = []
	for b in boundaryEdges:
		boundaryLengths.append(M.get_edge_length(b))
	# Get total length of the boundary
	totalBoundaryLength = sum(boundaryLengths)
	
	# Calculate sector angles
	sectorAngles = []
	for i in range(0, len(boundaryLengths)):
		angle = (2*math.pi*boundaryLengths[i]) / totalBoundaryLength
		sectorAngles.append(angle)

	# /// 1.1.1 formula (2)

	# Calculate UV positions on the circle
	uvPositions = []
	angleSum = 0
	# Get the smallest vertex and the edge it is part of
	minVert = M.get_edge(boundaryEdges[0])[0]
	minVertEdge = 0
	# Add the first UV position on (r,0) on the circle
	uvPositions.append((r, 0))
	boundEdgeCoords = []
	for i in range(1, len(sectorAngles)):
		angleSum = angleSum + sectorAngles[i-1]
		uv = r * (math.cos(angleSum), math.sin(angleSum))
		uvPositions.append(uv)
		if M.get_edge(boundaryEdges[i])[0] < minVert:
			minVert = M.get_edge(boundaryEdges[i])[0]
			minVertEdge = i
	# Get the smallest edge from all the edges
	smallestEdge = M.get_edge(boundaryEdges[minVertEdge])
	boundEdges = []
	for i in boundaryEdges:
		edge0 = M.get_edge(i)
		boundEdges.append(edge0)
	# Remove the smallest edge
	boundEdges.remove(smallestEdge)	
	walkPath = [smallestEdge]
	
	while minVert != smallestEdge[1]:
		for edge in boundEdges:
			if edge[0] == smallestEdge[1]:
				smallestEdge = edge
				walkPath.append(smallestEdge)
				boundEdges.remove(edge)
				break
	boundVerts = []
	for p in walkPath:
		boundVerts.append(p[0])
	nr_outer_vertices = len(boundVerts)

	for i in range(0, nr_outer_vertices):
		vertex = boundVerts[i]
		M.uv_coordinates[vertex] = uvPositions[i]		

	# /// 1.1.2 [weights calculated in formulas above]

	# /// 1.1.3 Laplacian System

	# Get the boundary edge with the vertex that has the lowest index,
	# also convert edge indices to tuples of vertex indices
	minIndex = 999999999999999
	walkEdge = 999999999999999
	boundEdgeCoords = []
	for i in boundaryEdges:
		smallestEdge = edges[i]
		boundEdgeCoords.append(smallestEdge)
		if smallestEdge[0] < minIndex:
			minIndex = smallestEdge[0]
			walkEdge = i

	# Get all the inner edges (||E_i||)
	E_i = [x for x in edges if x not in boundEdgeCoords]
	
	# /// 1.1.3 formula (3) (creating d0)

	# Create the tuples of the positions the of 1s and -1s 
	tuplesList = []
	for i in range(0, len(E_i)):
		tuplesList.append((i, E_i[i][0] , 1))
		tuplesList.append((i, E_i[i][1] , -1))

	# /// 1.1.3 formula (5)

	# Create sparse matrix W
	weightsList = []
	weightsCount = len(weights)
	for i in range(0, weightsCount):
		weightsList.append((i, i, weights[i]))
	W = ddm.Sparse_Matrix(weightsList, weightsCount, weightsCount)
	
	# Slice d0
	sliced_d0 = slice_triplets(tuplesList, boundVerts)
	d0_Blist, d0_Ilist = sliced_d0
	d0_I = ddm.Sparse_Matrix(d0_Ilist, len(E_i), len(V) - len(boundVerts))
	d0_B = ddm.Sparse_Matrix(d0_Blist, len(E_i), len(boundVerts))
	
	# Get u_B and v_B from the already calculated boundary UVs
	U_B = [u for (u,v) in uvPositions]
	V_B = [v for (u,v) in uvPositions]

	# -d0_I
	copyTuplesList = [(x, y, -z) for (x, y, z) in d0_Ilist]
	negd0_I = ddm.Sparse_Matrix(copyTuplesList, len(E_i), len(V) - len(boundVerts))

	# Right hand sides of formula (5) minus U or V
	transposenegd0_I = negd0_I.transposed()
	rhs = transposenegd0_I * W
	rhs = rhs * d0_B 

	# Left hand sides of formula (5) minus U or V
	lhs = d0_I.transposed() * W * d0_I
	# Calculate the the inner UVs
	lhs.Cholesky()
	U_i = lhs.solve(rhs * U_B)
	V_i = lhs.solve(rhs * V_B)
	# Create a list of tuples of all the UVs we have
	UV_i = list(zip(U_i, V_i))
	UV_b = list(zip(U_B, V_B))
	# Link inner vertices to UV coordinates
	UV_in = {}
	uvCount = 0
	for vert in range(0, len(V)):
		if vert not in boundVerts:
			UV_in[vert] = UV_i[uvCount]
			uvCount = uvCount + 1

	# Link boundary vertices to UV coordinates
	UV_bound = {}
	UV_bound[minIndex] = UV_b[0]
	uvCount = 1
	for i in range(1, len(UV_b)):
		nextCoord = edges[walkEdge][1]
		UV_bound[nextCoord] = UV_b[uvCount]
		walkEdge = edges.index([(u, v) for (u, v) in boundEdgeCoords if u == nextCoord][0])
		uvCount = uvCount + 1

	for i in range(0, len(V)):
		if i in boundVerts:
			M.uv_coordinates[i] = UV_bound[i]
		else:
			M.uv_coordinates[i] = UV_in[i]

	return M

# Using Least Squares Conformal Mapping, calculate the uv-coordinates of a given mesh M and return M with those uv-coordinates applied
def LSCM(M): 

	# TODO: implement yourself
	
	return M
	
# Builds a Mesh class object from the active object in the scene.
# in essence this function extracts data from the scene and returns it as a (simpler) Mesh class, triangulated where nessecary.
# This function is very similar to the get_vertices function but instead of triangle lists you should build a Mesh class instead.
def get_mesh():
	
	# Get the active object from the scene
	active_obj = bpy.context.active_object
	# Get its vertices
	vertices = []
	for i in range(0, len(active_obj.data.vertices)):
		vertices.append(active_obj.data.vertices[i].co)
	# Get its faces
	faces = []
	for face in active_obj.data.polygons:
		verts = face.vertices[:]
		faces.append(verts)

	return Mesh(vertices, faces)
	
# Given a Mesh class M, create a new object with name in the scene with the data from M
def show_mesh(M, name):
		
	# Note that in order to apply UV-coordinates to a mesh, the mesh needs to have at least 1 UV-layer (denoted as UV-map in the Blender interface under "data") with data.
	# You can then set the UV-coordinates of each loop (not vertex as in your own implemented Mesh class). The term "loop" is quite a misnomer in the Blender interface and 
	# differs from the use in the assignment itself as it simply means "some polyon in the mesh".
	# In order to view the UV-coordinates in the Blender interface make sure the renderer is set to "Blender Render", the mesh has a material (Properties -> Material) with a texture map 
	# (for example a Checkerboard under Properties -> Texture).
	# See https://docs.blender.org/api/current/bpy.types.Mesh.html#mesh-data for more details.
		
	# Create a mesh and object first
	mesh = bpy.data.meshes.new("mesh")
	obj = bpy.data.objects.new(name, mesh)
	# Link the object to the scene
	scene = bpy.context.scene
	scene.objects.link(obj)
	# Get mesh data
	vertices = M.get_vertices()
	faces = M.get_faces()
	# Add data to the mesh
	mesh.from_pydata(vertices, [], faces)
	# Update mesh changes
	mesh.update()
	# Create a b_mesh for getting a UV layer
	b_mesh = bmesh.new()
	b_mesh.from_mesh(mesh)
	# Create a UV layer
	uvLayer = b_mesh.loops.layers.uv.new()
	for triangle in b_mesh.faces:
		for loop in triangle.loops:
			uv = loop[uvLayer]
			uv.uv = M.get_uv_coordinate(loop.vert.index)
	# Convert the b_mesh back to a regular mesh
	b_mesh.to_mesh(mesh)
	mesh.update()



    