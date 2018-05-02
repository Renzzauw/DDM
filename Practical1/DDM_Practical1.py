# File created on: 2018-04-26 14:54:00.773449
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com

import bpy
from mathutils import Vector as Vector
from mathutils import Matrix as Matrix

# To view the printed output toggle the system console in "Window -> Toggle System Console"

# This function is called when the DDM operator is selected in Blender.
def DDM_Practical1(context):
	print("yeet")
	# TODO: get the triangles for the active mesh and use show_mesh() to display a copy
	show_mesh(get_triangles(context))


	# TODO: print the euler_characteristic value for the active mesh
	
	# TODO: print whether the active mesh is closed
	
	# TODO: print whether the active mesh is orientable
	
	# TODO: print the genus of the active mesh
	
	# TODO: use show_mesh() to display the maximal_independent_set of the dual of the active object
	
	
	print(get_triangles(context))

# Builds a mesh using a list of triangles
def show_mesh(triangles):
	# Create a mesh and object first
	mesh = bpy.data.meshes.new("mesh")
	obj = bpy.data.objects.new("newObject", mesh)
	# Link the object to the scene
	scene = bpy.context.scene
	scene.objects.link(obj)
	# Add the vertices of all triangles to a list
	vertices = []
	# Create a list for the faces too
	i = 0
	faces = []
	for t in triangles:
		# Create a face (list of vertices of the triangle) and increment the index of the vertices
		face = []
		face.append(i)
		i += 1
		face.append(i)
		i += 1
		face.append(i)
		i += 1
		# Add the face to the list of faces
		faces.append(face)
		# Add the vertices to the list of vertices
		vertices.append(t[0])
		vertices.append(t[1])
		vertices.append(t[2])
	
	# add data to the mesh
	mesh.from_pydata(vertices, [], faces)
	# Update mesh changes
	mesh.update()
	
# Returns the faces of the active object as a list of triplets of points
def get_triangles(context):
	return [(Vector([0, 0, 0]), Vector([1, 0, 0]), Vector([0, 1, 0])), (Vector([1, 0, 0]), Vector([1, 1, 0]), Vector([0, 1, 0]))]

# Calculates the Euler characteristic of the given list of triangles
def euler_characteristic(triangles):
	return 0

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
	
