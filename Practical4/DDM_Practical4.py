# File created on: 2018-04-26 14:54:00.776450
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com

import ddm
from mathutils import Vector

# Place additional imports here

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
	# The indices of the edges should correspond to the locations of the weights in your weight calculation.
	# All subsequent calls that do something with edges should return their indices. Getting the actual edge can then be done by calling get_edge(index).
	def build_edge_list(self):
	
		# TODO: implement yourself ***DONE***

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
			if tuple1 not in result or tuple1mirror not in result:
				result.append(tuple1)
			if tuple2 not in result or tuple2mirror not in result:
				result.append(tuple2)
			if tuple3 not in result or tuple3mirror not in result:
				result.append(tuple3)	



		print("Build_edge_list: ", result)

		self.edges = result #[ (0, 1), (1, 2) ]
	
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
	
		# TODO: implement yourself ***DONE***
		
		# Get the list of all edges in the Mesh
		edges = self.get_edges()
		# Get each edge from the given face
		edge1 = (face[0], face[1])
		edge2 = (face[1], face[2])
		edge3 = (face[2], face[0])
		edge1mirror = (face[1], face[0])
		edge2mirror = (face[2], face[1])
		edge3mirror = (face[0], face[2])

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
	
	# Returns the flap of the given edge belonging to edge_index, that is two faces connected to the edge 1 for a boundary edge, 2 for internal edges
	def get_flaps(self, edge_index):
	
		# TODO: implement yourself
	
		# Watch out: edges might be on the boundary
	
		return [ (0, 1, 2), (0, 3, 4) ]
		
	# Returns the length of the given edge with edge_index
	def get_edge_length(self, edge_index):
		
		# TODO: implement yourself
		
		return 0
		
	# Returns whether the edge has two adjacent faces
	def is_boundary_edge(self, edge_index):
		
		# TODO: implement yourself
		
		return False
	
	# Returns the boundary of the mesh by returning the indices of the edges (from the internal edge list) that lie around the boundary.
	def boundary_edges(self):
		
		# TODO: implement yourself
		
		return False
		
	# Place any other accessors you need here
	
# This function is called when the DDM operator is selected in Blender.
def DDM_Practical4(context):
	
	# TODO: remove example code and implement Practical 4
	
	# Example mesh construction
	
	# The vertices are stored as a list of vectors (ordered by index)
	vertices = [Vector( (0,0,0) ), Vector( (0,1,0) ), Vector( (1,1,0) ), Vector( (1,0,0) )]
	
	# The faces are stored as triplets (triangles) of vertex indices, polygons need to be triangulated as in previous practicals. This time edges should also be extracted.
	faces = [ (0, 1, 2), (0, 3, 4) ]
	
	# Construct a mesh with this data
	M = Mesh(vertices, faces)
	
	# You can now use the accessors to access the mesh data
	#print(M.get_edges())
	
	# An example of the creation of sparse matrices
	A = ddm.Sparse_Matrix([(0, 0, 4), (1, 0, 12), (2, 0, -16), (0, 1, 12), (1, 1, 37), (2, 1, -43), (0, 2, -16), (1, 2, -43), (2, 2, 98)], 3, 3)
	
	# Sparse matrices can be multiplied and transposed
	B = A.transposed() * A
	
	# Cholesky decomposition on a matrix
	B.Cholesky()
	
	# Solving a system with a certain rhs given as a list
	rhs = [2, 2, 2]
	
	x = B.solve(rhs);
	
	# A solution should yield the rhs when multiplied with B, ( B * x - v should be zero)
	print(Vector( B * x ) - Vector(rhs) )
	
	# You can drop the matrix back to a python representation using 'flatten'
	print(B.flatten())
	
	# TODO: show_mesh on a copy of the active mesh with uniform UV coordinates, call this mesh "Uniform"
	
	# TODO: show_mesh on a copy of the active mesh with cot UV coordinates, call this mesh "Cot"
	
	# TODO: show_mesh on a copy of the active mesh with boundary free UV coordinates, call this mesh "LSCM"
	

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

# Returns the weights for each edge of mesh M.
# It therefore consists of a list of real numbers such that the index matches the index of the edge list in M.
def cotan_weights(M, r):
	
	# TODO: implement yourself
	
	pass
	
# Same as above but for uniform weights
def uniform_weights(M, r):

	# TODO: implement yourself

	pass
	
# Given a set of weights, return M with the uv-coordinates set according to the passed weights
def Convex_Boundary_Method(M, weights):
	
	# TODO: implement yourself
	
	return M

# Using Least Squares Conformal Mapping, calculate the uv-coordinates of a given mesh M and return M with those uv-coordinates applied
def LSCM(M):

	# TODO: implement yourself

	return M
	
# Builds a Mesh class object from the active object in the scene.
# in essence this function extracts data from the scene and returns it as a (simpler) Mesh class, triangulated where nessecary.
def get_mesh():

	# TODO: implement yourself

	return Mesh([], [])
	
# Given a Mesh class M, create a new object with name in the scene with the data from M
def show_mesh(M, name):
	
	# TODO: implement yourself
	
	pass