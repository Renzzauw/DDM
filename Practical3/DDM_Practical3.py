# File created on: 2018-04-26 14:54:00.775450
#
# MADE BY:
# - 5964962 Renzo Schindeler
# - 5845866 Kasper Nooteboom
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

# To view the printed output toggle the system console in "Window -> Toggle System Console"

def mesh_from_array(A, n):
	
	vertices = []
	print(A)
	for y in range(0, n - 1):
		for x in range(0, n - 1):
			current = A[y * n + x]
			up = A[(y + 1) * n + x]
			right = A[ (y * n) + (x + 1)]
			rightup = A[ ((y + 1) * n) + (x + 1)]
			t1 = [current, right, up]
			t2 = [right, rightup, up]
			vertices.append(t1)
			vertices.append(t2)



	return vertices
	
def De_Casteljau(A, n, s):
	
	if (s == 0):
		return A

	numberOfPoints = (n - 1) * s + n

	segmentSize = (1/(n-1))/(s+1)

	points = []

	for i in range(1, numberOfPoints - 1):
		point = i * segmentSize
		points.append(point)

	segmentList = []
	for x in range(0,n):
		lijst = []
		for y in range(0,n):
			lijst.append(A[x+n*y])
		segmentList.append(lijst)	
		
	puntenLijst = []
	for i in n:
		puntenLijst.append(A[i])
	
	for t in range(0, len(points)):
		for i in segmentList:
			p = F(i)
			puntenLijst.append(p)

	for i in range(len(A)-n, len(A)):
		puntenLijst.append(A[i])

	

	return []

def F(C, t):
	while len(C) > 1:
		C = CasteljauStep(C, t)

	return C[0]

def CasteljauStep():


def control_mesh(n, length):
	# Create an array for the vertices
	vertices = []
	# Calculate the distance between each vertex
	distance = length / (n - 1)
	# Generate the vertices sorted by y and then x
	# The z-position is generated randomly
	for y in range(0, n):
		for x in range(0, n):
			xpos = x * distance
			ypos = y * distance
			zpos = numpy.random.random_sample() * 0.5
			vertex = (xpos, ypos, zpos)
			vertices.append(vertex)
	
	return vertices
	
def line_intersect(A, n, p1, p2, e):
	return False
	
def subdivisions(n, s):
	return 1
	
def DDM_Practical3(context):
	
	n = 10
	length = 1
	s = 3
	
	A = control_mesh(n, length)
	B = De_Casteljau(A, n, s)
	
	# TODO: Calculate the new size of the subdivided surface
	n_B = subdivisions(1, s)
	
	show_mesh(mesh_from_array(B, n_B))
	#show_mesh(mesh_from_array(A, n))

	p1 = (1,2,3)
	p2 = (3,4,5)
	
	print(line_intersect(A, n, p1, p2, 0.01))
	
# Builds a mesh using a list of triangles
# This function is the same as the previous practical
def show_mesh(triangles):
	# Create a mesh and object first
	mesh = bpy.data.meshes.new("mesh")
	obj = bpy.data.objects.new("Mesh", mesh)
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