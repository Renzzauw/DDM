# File created on: 2018-06-12 11:39:18.938293
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
import bpy
import numpy
import math
from numpy import array as Vector
from numpy import matrix as Matrix
from numpy import identity

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
		v1 = edge[0]
		v2 = edge[1]
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

# Create a global mesh object M
M = Mesh([],[])
d0 = []
d0_I = ddm.Sparse_Matrix([], 1, 1)
d0_B = ddm.Sparse_Matrix([], 1, 1)
neg_d0_I = ddm.Sparse_Matrix([], 1, 1)
obj = ""
handles = []
handleVerts = []
W = ddm.Sparse_Matrix([], 1, 1)
E_i = []
edgesGlob = []
lhs = ddm.Sparse_Matrix([], 1, 1)
neighborlist = []
stopARAP = False

# Return a list of vertices
def get_vertices(context):

	# Get the currently active object
	obj = context.scene.objects.active
	# Create empty triangles list
	vertices = []
	# Get this object's polygons
	polygons = obj.data.polygons
	# Get all the vertices from each mesh polygon
	for p in polygons:
		polygonVertices = p.vertices
		for i in range(0, len(polygonVertices)):
			if obj.data.vertices[polygonVertices[i]].co not in vertices:
				vertices.append(obj.data.vertices[polygonVertices[i]].co)

	return vertices
	
# Returns a list of triangles of vertex indices (you need to perform simple triangulation) 
def get_faces(context):	

	# Get the currently active object
	obj = context.scene.objects.active
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
			triangles.append(tuple(tri))
		
	return triangles

# Returns the 1-ring (a list of vertex indices) for a vertex index
def neighbor_indices(vertex_index, vertices, faces):
	neighbors = set()
	for face in faces:
		if vertices.index(face[0]) == vertex_index:
			neighbors.add(vertices.index(face[1]))
			neighbors.add(vertices.index(face[2]))
		if vertices.index(face[1]) == vertex_index:
			neighbors.add(vertices.index(face[0]))
			neighbors.add(vertices.index(face[2]))
		if vertices.index(face[2]) == vertex_index:
			neighbors.add(vertices.index(face[0]))
			neighbors.add(vertices.index(face[1]))

	return list(neighbors)
	
# Calculates the source (non-sparse) matrix P
def source_matrix(p_index, vertices, neighbor_indices):
	pDist = []
	currPoint = vertices[p_index]
	# Subtract each neighbour from the current point
	for p in neighbor_indices:
		diff = currPoint - vertices[p]
		pDist.append(diff)

	return Matrix(pDist)
	
# Calculates the target (non-sparse) matrix Q
def target_matrix(p_index, vertices, neighbor_indices):
	V = list(vertices)
	global handleVerts
	allVerts = M.get_vertices()
	for v in handleVerts:
		V.append(allVerts[v])
	pDist = []
	currPoint = V[p_index]
	# Subtract each neighbour from the current point
	for p in neighbor_indices:
		try:
			diff = currPoint - V[p]
			pDist.append(diff)
		except AttributeError:
			print('vertices doet kut bij:', p)
			print(V)
			raise Exception('en stop maar')

	return Matrix(pDist)
	
# Returns a triple of three dense matrices that are the Singular Value Decomposition (SVD)
def SVD(P, Q):

	# Calculate S
	S = P.transpose() * Q
	U, Sigma, V = numpy.linalg.svd(S)

	# Make sure that the result of the singular value decomposition is a triple of numpy.matrix and not some other datastructure.
	return (U, Sigma, V)

# Returns the dense matrix R
def rigid_transformation_matrix(U, Sigma, V):
	# U * transpose of V
	Ri = U * V.transpose()
	# Calculate the determinant of Ri
	det = numpy.linalg.det(Ri)
	# Check if determinant == -1
	if det == -1:
		smallestValue = 99999999999
		smallestIndex = 0
		for i in range(0, Sigma.shape[0]):
			if Sigma[i] < smallestValue:
				smallestValue = Sigma[i][i]
				smallestIndex = i
		# Calculate the new Ri
		newSigma = numpy.identity(Sigma.shape[0])
		newSigma[smallestIndex][smallestIndex] = -1
		Ri = U * newSigma * V.transpose()

	return Ri

# Returns a list of rigid transformation matrices R, one for each vertex (make sure the indices match)
# the list_of_1_rings is a list of lists of neighbor_indices
def local_step(source_vertices, target_vertices, list_of_1_rings):
	result = []
	for i in range(0, len(list_of_1_rings)):
		# (1) Compose source and traget matrices
		P = source_matrix(i, source_vertices, list_of_1_rings[i])
		Q = target_matrix(i, target_vertices, list_of_1_rings[i])

		# (3) Decompose S using SVD
		U, Sigma, V = SVD(P, Q)

		# (4) Calculate Ri
		Ri = rigid_transformation_matrix(U, Sigma, V)
		result.append(Ri)

	return result

# Returns the triplets of sparse d_0 matrix
def d_0(vertices, faces):
	global E_i

	# Create the tuples of the positions the of 1s and -1s 
	tuplesList = []
	for i in range(0, len(E_i)):
		tuplesList.append((i, vertices.index(E_i[i][0]), 1))
		tuplesList.append((i, vertices.index(E_i[i][1]), -1))

	return tuplesList
	
# Return the sparse diagonal weight matrix W
def weights(vertices, faces):
	
	global M
	global W
	global E_i
	global edgesGlob

	# Get all edges
	edges = edgesGlob

	weights = []
	for edge in range(0, len(edges)):
		if edges[edge] not in E_i:
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
	
	# Create sparse matrix W
	weightsList = []
	weightsCount = len(weights)
	for i in range(0, weightsCount):
		weightsList.append((i, i, weights[i]))
	Whiephoi = ddm.Sparse_Matrix(weightsList, weightsCount, weightsCount)
	
	return Whiephoi

# Function for triplets list × -1
def negTripList(triplets):
	copyList = [(x, y, -z) for (x, y, z) in triplets]
	return copyList
	
# Returns the right hand side of least-squares system
def RHS(vertices, rigids):
	global d0
	global d0_I
	global d0_B
	global neg_d0_I
	global handles
	global handleVerts
	global W
	global E_i
	global edgesGlob

	# Compute G
	g = []
	countertje = 0
	for edge in E_i:
		diff = edge[0] - edge[1]
		rid1 = rigids[vertices.index(edge[0])]
		rid2 = rigids[vertices.index(edge[1])]
		doemaaroptellen = numpy.add(rid1, rid2)
		doemaardivide = numpy.divide(doemaaroptellen, 2)
		dinges = numpy.matmul(doemaardivide, diff)
		g.append( (countertje, 0, dinges[0]) )
		g.append( (countertje, 1, dinges[1]) )
		g.append( (countertje, 2, dinges[2]) )
		countertje = countertje + 1
	G = ddm.Sparse_Matrix(g, len(E_i), 3)

	# Compute p_appelstroop_B
	appelstrooplijst = []
	countertje = 0
	for hendel in handles:
		for hendelpunt in hendel[0]:
			hendelpunt = vertices[hendelpunt]
			hendelpunt = Vector((hendelpunt[0], hendelpunt[1], hendelpunt[2], 1))
			wowgeinig = numpy.matmul(hendel[1], hendelpunt)
			appelstrooplijst.append( (countertje, 0, wowgeinig[0]) )
			appelstrooplijst.append( (countertje, 1, wowgeinig[1]) )
			appelstrooplijst.append( (countertje, 2, wowgeinig[2]) )
			countertje = countertje + 1
	p_appelstroop_B = ddm.Sparse_Matrix(appelstrooplijst, len(handleVerts), 3)

	linkie = neg_d0_I.transposed() * W * d0_B * p_appelstroop_B
	rechtie = d0_I.transposed() * G

	rhs = numpy.add(linkie.flatten(), rechtie.flatten())

	return rhs

	# You need to convert the end result to a dense vector
	# return Vector([1,2,4,5,5,6])
	# lol why
	
# Returns a list of vertices coordinates (make sure the indices match)
def global_step(vertices, rigid_matrices):
	global lhs

	# TODO: Construct RHS
	rhs = RHS(vertices, rigid_matrices)
	rightx = []
	righty = []
	rightz = []
	for v in rhs:
		rightx.append(v[0])
		righty.append(v[1])
		rightz.append(v[2])
	
	# TODO: solve separately by x, y and z (use only a single vector)
	x = lhs.solve(rightx)
	y = lhs.solve(righty)
	z = lhs.solve(rightz)

	# Convert to coordinates
	augurk = []
	for i in range(len(x)):
		ding = numpy.array([x[i], y[i], z[i]]) #Vector((x[i], y[i], z[i]))
		print(type(ding))
		print(ding)
		augurk.append(ding)

	return augurk
	
# Returns the left hand side of least-squares system
def precompute(vertices, faces):
	global d0
	global d0_I
	global d0_B
	global neg_d0_I
	global W
	global E_i

	# TODO: construct d_0 and split them into d_0|I and d_0|B
	d0 = d_0(vertices, faces)

	# Get inner vertices
	innerVerts = []
	for v in range(len(vertices)):
		if v not in handleVerts:
			innerVerts.append(v)

	# Get all necessary splits of d0
	d0_Ilist, d0_Blist = slice_triplets(d0, innerVerts) 
	neg_d0_Ilist = negTripList(d0_Ilist)
	neg_d0_I = ddm.Sparse_Matrix(neg_d0_Ilist, len(E_i), len(innerVerts))
	d0_I = ddm.Sparse_Matrix(d0_Ilist, len(E_i), len(innerVerts))
	d0_B = ddm.Sparse_Matrix(d0_Blist, len(E_i), len(vertices) - len(innerVerts))

	# TODO: construct LHS with the elements above and Cholesky it
	lhs = d0_I.transposed() * W * d0_I
	lhs.Cholesky()

	return lhs

# Initial guess, returns a list of identity matrices for each vertex
def initial_guess(vertices):
	doenouesnormaal = []
	for v in vertices:
		doenouesnormaal.append(numpy.identity(3))
	return doenouesnormaal
	
# Returns the new target vertices as a list (make sure the indices match)
def ARAP_iteration(vertices, target_vertEces, max_movement):
	global stopARAP
	global neighborlist

	# TODO: local step
	Ri = local_step(vertices, target_vertEces, neighborlist)
	
	# TODO: global step
	newTarget = global_step(vertices, Ri)

	# Checks if the tolerance has been fucked with
	maxDist = 0
	for v in range(len(newTarget)):
		distVect = newTarget[v] - vertices[v]
		dist = math.sqrt(distVect[0]**2 + distVect[1]**2 + distVect[2]**2)
		if maxDist < dist:
			maxDist = dist
	if maxDist < max_movement:
		stopARAP = True

	return newTarget
	
def DDM_Practical5(context):
	print("######################################################################################################")
	max_movement = 0.001
	max_iterations = 100
	global M
	global W
	global obj
	global handles
	global handleVerts
	global E_i
	global edgesGlob
	global lhs
	global neighborlist

	# Get some basic information
	obj = context.scene.objects.active
	vertices = get_vertices(context)
	faces = get_faces(context)
	M = Mesh(vertices, faces)
	edgesGlob = M.get_edges()

	# Setup the list of neighbors
	for v in range(len(vertices)):
		neighborlist.append(neighbor_indices(v, vertices, faces))

	# TODO: get handles
	handles = get_handles(obj)
	handleVertsLists = [x for (x, y) in handles]
	handleVerts = list(set([item for sublist in handleVertsLists for item in sublist]))
	
	# TODO: get mesh data
	edges = edgesGlob
	boundEdges = []
	for i in range(len(edges)):
		edge = edges[i]
		if vertices.index(edge[0]) in handleVerts and vertices.index(edge[1]) in handleVerts:
			boundEdges.append(i)
	E_i = [edge for edge in edges if edges.index(edge) not in boundEdges]
	
	W = weights(vertices, faces)
	
	# TODO: precompute
	lhs = precompute(vertices, faces)
	
	# TODO: initial guess
	tgt = global_step(vertices, initial_guess(vertices))

	# TODO: ARAP until tolerance
	countor = 1
	for i in range(0, max_iterations):
		if stopARAP: break
		print('RONDE:::::::::::::::::::D', countor)
		tgt = ARAP_iteration(vertices, tgt, max_movement)
		countor = countor + 1

	
#########################################################################
# You may place extra variables and functions here to keep your code tidy
#########################################################################
	
# Find the vertices within the bounding box by transforming them into the bounding box's local space and then checking on axis aligned bounds.
def get_handle_vertices(vertices, bounding_box_transform, mesh_transform):

	result = []

	# Fibd the transform into the bounding box's local space
	bounding_box_transform_inv = bounding_box_transform.copy()
	bounding_box_transform_inv.invert()
	
	# For each vertex, transform it to world space then to the bounding box local space and check if it is within the canonical cube x,y,z = [-1, 1]
	for i in range(len(vertices)):
		vprime = vertices[i].co.copy()
		vprime.resize_4d()
		vprime = bounding_box_transform_inv * mesh_transform * vprime
		
		x = vprime[0]
		y = vprime[1]
		z = vprime[2]
		
		if (-1 <= x) and (x <= 1) and (-1 <= y) and (y <= 1) and (-1 <= z) and (z <= 1):
			result.append(i)

	return result

# Returns the local transform of the object
def get_transform_of_object(name):
	return bpy.data.objects[name].matrix_basis
	
# Finds the relative transform from matrix M to T
def get_relative_transform(M, T):
	
	Minv = M.copy()
	Minv.invert()
		
	return T * Minv
	
# Returns a list of handles and their transforms
def get_handles(source):
	
	result = []
	
	mesh_transform = get_transform_of_object(source.name)
	
	# Only search up to (and not including) this number of handles
	max_handles = 10
	
	# For all numbered handles
	for i in range(max_handles):
	
		# Construct the handles representative name
		handle_name = 'handle_' + str(i)
		
		# If such a handle exists
		if bpy.data.objects.get(handle_name) is not None:
			
			# Find the extends of the aligned bounding box
			bounding_box_transform = get_transform_of_object(handle_name)
			
			# Interpret the transform as a bounding box for selecting the handles
			handle_vertices = get_handle_vertices(source.data.vertices, bounding_box_transform, mesh_transform)
			
			# If a destination box exists
			handle_dest_name = handle_name + '_dest'
			if bpy.data.objects.get(handle_dest_name) is not None:
				
				bounding_box_dest_transform = get_transform_of_object(handle_dest_name)
				
				result.append( (handle_vertices, get_relative_transform(bounding_box_transform, bounding_box_dest_transform) ) ) 
				
			else:
			
				# It is a rigid handle
				m = Matrix([])
				m = identity(4)
				result.append( (handle_vertices, m) )
			
	return result

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