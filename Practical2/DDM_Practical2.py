# File created on: 2018-04-26 14:54:00.774450
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com

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
	
	# This function returns the result of estimated function f(q) which is essentially the entire estimation plus the calculation of its result, note that the estimated polynomial is different for every q = (x, y, z)
	def sample(self, x, y, z):
		pass

# This function is called when the DDM Practical 2 operator is selected in Blender.
def DDM_Practical2(context):
	pass
	
#########################################################################
# You may place extra variables and functions here to keep your code tidy
#########################################################################

# Returns the points of the first object
def get_vertices(context):
	result = []
	for vertex in context.active_object.data.vertices:
		result.append( tuple(vertex.co) )

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

# The vector containing the values for 'c_m'
def constraint_points(points, normals, epsilon, radius):
	pass

# The vector 'd'
def constraint_values(points, normals, epsilon, radius):
	pass
	
# The vector (NOT matrix) 'W'
def weights(q, constraints):
	pass

# The vector that contains the numerical values of each term of the polynomial, this is NOT vector 'a'
def indeterminate(q, degree):
	return polyvander3d(q[0], q[1], q[2], [degree, degree, degree] )[0]

# Returns 'C'
def MatrixC(q, points, normals, epsilon, radius):
	pass

# Returns the Wendland weight for a given distance
def Wendland(distance):
	pass

# Returns the distance between vector 'a' and 'b'
def distance(a, b):
	return (b - a).length

# Builds a mesh using a list of triangles
# This function is the same as from the previous practical
def show_mesh(triangles):
	pass
