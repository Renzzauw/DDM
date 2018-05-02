# File created on: 2018-04-26 14:54:00.775450
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
	return []
	
def De_Casteljau(A, n, s):
	return []

def control_mesh(n, length):
	return []
	
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
	
	p1 = (1,2,3)
	p2 = (3,4,5)
	
	print(line_intersect(A, n, p1, p2, 0.01))
	
# Builds a mesh using a list of triangles
# This function is the same as the previous practical
def show_mesh(triangles):
	pass