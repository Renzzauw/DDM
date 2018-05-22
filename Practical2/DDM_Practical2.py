# File created on: 2018-05-08 15:24:47.263919
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
import sys
import math
from numpy.polynomial.polynomial import polyvander3d
from mathutils import Vector
from numpy import empty as new_Matrix
from numpy import matrix as Matrix
from math import sqrt

# You can perform the marching cubes algorithm by first constructing an instance of the marching cubes class:
#
# mc = My_Marching_Cubes()
#
# and then calling the calculate function for this class:
#
# mc.calculate(origin_x, origin_y, origin_z, x_size, y_size, z_size, cube_size)
#
# Where:
# - (origin_x, origin_y, origin_z) is the bottom left coordinate of the area meshed by marching cubes
# - (x_size, y_size, z_size) is the number of cubes in each direction
# - cube_size is the length of the edge of a single cube
#

class My_Marching_Cubes(ddm.Marching_Cubes):
	# You may place extra members and functions here
	
	def __init__(self, points, normals, epsilon, radius, wendland_constant, degree):
		super().__init__()
		
		# Build a dictionary for getting the normals
		self.point_normal = {}
		for i in range(len(points)):
			self.point_normal[ (points[i][0], points[i][1], points[i][2]) ] = normals[i]
		
		# Do NOT modify these parameter values within this class
		self.epsilon = epsilon
		self.radius = radius
		self.degree = degree
		self.wendland_constant = wendland_constant
		
		# TODO: construct a ddm.Grid *** DONE (niet getest)*** 
		# Use this grid for queries
		self.grid = ddm.Grid([point.to_tuple() for point in points])
	
	# Returns the normals belonging to a given (sub)set of points
	def normals_for_points(self, points):
		result = []
		for point in points:
			result.append(self.point_normal[(point[0], point[1], point[2])])
		return result
	
	# Queries the grid for all points around q within radius
	def query_points(self, q, radius):
		result = []
		for point in self.grid.query(q, radius):
			result.append(Vector(point))
		return result
	
	# This function returns the result of estimated function f(q) which is essentially the entire estimation plus the calculation of its result, note that the estimated polynomial is different for every q = (x, y, z)
	def sample(self, x, y, z):
		
		# TODO: make sure that your radial query always contains enough points to a) ensure that the MLS is well defined, b) you always know if you are on the inside or outside of the object.
		


		return distance(Vector([0, 0, 0]), Vector([x, y, z])) - 1
		
		
# This function is called when the DDM Practical 2 operator is selected in Blender.
def DDM_Practical2(context):

	# TODO: modify this function so it performs reconstruction on the active point cloud
	
	points = []
	normals = []
	epsilon = get_epsilon(points)
	radius = get_radius(points)
	degree = get_degree()
	
	mc = My_Marching_Cubes(points, normals, epsilon, radius, degree)
	
	triangles = mc.calculate(-1, -1, -1, 20, 20, 20, 0.1)
	
	show_mesh(triangles)

	
#########################################################################
# You may place extra variables and functions here to keep your code tidy
#########################################################################


# Returns the points of the first object
def get_vertices(context):
	result = []
	for vertex in context.active_object.data.vertices:
		result.append( Vector(vertex.co) )

	return result

# Returns the normals of the first object
def get_normals(context):
	if ('surface_normals' not in context.active_object):
		print("\n\n##################\n\nWARNING:\n\nThis object does not contain any imported surface normals! Please use one of the example point clouds provided with the assignment.\n\n##################\n\n")
		return []

	result = []

	for normal in context.active_object['surface_normals']:
		result.append( Vector( [normal[0], normal[1], normal[2] ] ) )

	return result

# Returns an query radius for the given point set
def get_radius(points):
	
	# TODO: Implement 

	radius = 0
	

	print("Radius: ", radius)
	return radius

# Returns the epsilon for the given point set
def get_epsilon(points):
	
	# TODO: Implement ***DONE (niet getest)***
	maxDist = 0

	# Get the amount of points
	length = len(points)
	# Get the largest distance between any of the 2 vectors
	for i in range(0, length):
		for j in range(0, length):
			maxDist = max(distance(i, j), maxDist)

	# As the assignment states, multiply it by 0.01
	epsilon = 0.01 * maxDist
	
	print("Epsilon: ", epsilon)
	return epsilon
	
# Returns the degree 'k' used in this reconstruction
def get_degree():
	return 2
	
# Returns the minimum and the maximum corner of a point set
def bounding_box(points):
	
	# TODO: implement ***DONE (niet getest)***

	minX = points[0][0]
	minY = points[0][1]
	minZ = points[0][2]
	maxX = points[0][0]
	maxY = points[0][1]
	maxZ = points[0][2]

	# Get the minimal and maximal x, y and z values
	for i in points:
		minX = min(minX, points[i][0])
		minY = min(minY, points[i][1])
		minZ = min(minZ, points[i][2])

		maxX = max(maxX, points[i][0])
		maxY = max(maxY, points[i][1])
		maxZ = max(maxZ, points[i][2])

	# Create the vectors
	minVec = Vector[minX, minY, minZ]
	maxVec = Vector[maxX, maxY, maxZ]

	# Log the minimum and maximum bounding box points
	print("Bounding Box points, min: ", minVec, " max: ", maxVec )

	# Return these corners
	return (minVec, maxVec)
	
# The vector containing the values for '{c_m}'
def constraint_points(points, normals, epsilon, radius):
	# Column vector C volgens opdracht
	# TODO: Implement
	

	# C = points + constraint points (normale points +- epsilon * normal) die binnen radius vallen

	return [Vector([0, 0, 0])]

# The vector 'd'
def constraint_values(points, normals, epsilon, radius):
	
	# TODO: Implement
	
	return [0]
	
# The vector (NOT matrix) 'W'
def weights(q, constraints, wendland_constant):
	
	# TODO: Implement
	
	return [0]

# The vector that contains the numerical values of each term of the polynomial, this is NOT vector 'a'
def indeterminate(q, degree):
	return polyvander3d(q[0], q[1], q[2], [degree, degree, degree] )[0]

# For a given list of coefficients a, use this to find the actual value for 'f(p)'
def polynomial(p, a, degree):
	
	# TODO: Implement
	
	return 0
	
# Returns 'C'
# NOTE: There is no need for this function to be passed the parameters 'wendland_constant', 'epsilon' and 'radius', you can structure the assignment in such a way this is not necessary.
def MatrixC(q, constraints, degree):	
	
	# TODO: Implement
	
	return new_Matrix([1, 1])
	
# Returns the Wendland weight for a given distance with shape/range parameter wendland_constant
def Wendland(distance, wendland_constant):
	
	# TODO: Implement ***DONE (niet getest)***

	weight = ((1 - (distance/wendland_constant))**4) * (4*(distance/wendland_constant)+1)

	
	return weight

# Returns the distance between vector 'a' and 'b'
def distance(a, b):
	return (b - a).length

# Builds a mesh using a list of triangles
# This function is the same as from the previous practical
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

#########################################################################
# Extra functions we have created
#########################################################################
