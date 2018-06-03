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
	denominator = numberOfPoints - 1

	# Zet alle punten in A om in lijsten van punten in y-richting met grootte n
	lijstenInYRichting = []
	for x in range(0,n):
		lijst = []
		for y in range(0,n):
			lijst.append(A[y*n+x])	
		lijstenInYRichting.append(lijst)	

	pointsAfterYSubDiv = []

	# Ga voor iedere lijst in y-richting subdividen
	for i in lijstenInYRichting:
		lijst = i
		for t in range(0, numberOfPoints):
			point = CasteljauStep(lijst, t/denominator)
			pointsAfterYSubDiv.append(point)

	# Zet alle punten in A om in lijsten van punten in x-richting met grootte n
	lijstenInXRichting = []
	for x in range(0, numberOfPoints):
		lijst = []
		for y in range(0, n):
			lijst.append(pointsAfterYSubDiv[y*numberOfPoints+x])
		lijstenInXRichting.append(lijst)	

	pointsAfterXSubDiv = []

	# Ga voor iedere lijst in x-richting subdividen
	for i in lijstenInXRichting:
		lijst = i
		for t in range(0, numberOfPoints):
			point = CasteljauStep(lijst, t/denominator)
			pointsAfterXSubDiv.append(point)

	return pointsAfterXSubDiv
	
# Voer De Casteljau uit op een gegeven breuk en return het bijbehorende punt
def CasteljauStep(C, t):
	points = C
	while len(points) > 1:
		newPoints = []
		for i in range(0, len(points)-1):
			pos = ((points[i][0] + t * (points[i+1][0]-points[i][0])), (points[i][1] + t * (points[i+1][1]-points[i][1])), (points[i][2] + t * (points[i+1][2]-points[i][2])))
			newPoints.append(pos)
		points = newPoints

	return points[0]


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
			if (x == 0) or (y == 0) or (x == n - 1) or (y == n - 1):
				zpos = 0
			else:
				zpos = numpy.random.random_sample() * 3 - 1.5
			vertex = (xpos, ypos, zpos)
			vertices.append(vertex)
	
	return vertices
	
def line_intersect(A, n, p1, p2, e):
	return False
	
def subdivisions(n, s):
	return (n - 1) * s + n
	
def DDM_Practical3(context):
	
	n = 10
	length = 1
	s = 3
	
	A = control_mesh(n, length)
	B = De_Casteljau(A, n, s)
	
	n_B = subdivisions(n, s)
	
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