# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def is_array(var):
	return isinstance(var, (list, tuple))

def generate_random_polynomial():
	#both x and y are defined by two polynomials in a third variable p, plus
	#an initial angle (that, when connecting splines, will be the same as
	#the final angle of the previous polynomial)
	#Both polynomials are third order.
	#The polynomial for x is aU, bU, cU, dU
	#The polynomial for y is aV, bV, cV, dV
	#aU and bU are always 0 (start at origin) and bV is always 0 (derivative at
	#origin is 0). bU must be positive
	# constraints initial coordinates must be the same as
	# ending coordinates of the previous polynomial
	aU = 0
	aV = 0
	# initial derivative must the same as the ending
	# derivative of the previous polynomial
	bU = (10-6)*np.random.random()+6  #around 8
	bV = 0
	#we randonmly generate values for cU and dU in the range ]-1,1[
	cU = 2*np.random.random()-1
	dU = 2*np.random.random()-1
	finalV = 10*np.random.random()-5
	#final derivative between -pi/6 and pi/6
	finald = np.tan((np.pi/3)*np.random.random() - np.pi/6)
	#now we fix parameters to meet the constraints:
	#bV + cV + dV = finalV 
	#angle(1) = finald; see the definition of angle below
	Ud = bU + 2*cU + 3*dU
	#Vd = bU + 2*cU + 3*dU = finald*Ud
	dV = finald*Ud - 2*finalV + bV
	cV = finalV - dV - bV
	return ((aU,bU,cU,dU), (aV,bV,cV,dV))
	
def angle(p, U, V):
	Ud = derivative(p,U)
	Vd = derivative(p,V)
	return (np.arctan(Vd/Ud)) if abs(Ud) > abs(Vd/1000) else (np.pi/2)
		
def poly(p, points):
	return points[0] + points[1]*p + points[2]*p**2 + points[3]*p**3
	
def derivative(p, points):
	return points[1] + 2*points[2]*p + 3*points[3]*p**2
			
def euclidean_distance(a,b):
	return np.sqrt(sum((j-k)**2 for (j,k) in zip(a,b)))
	
def rotate(x,y,theta):
	return (x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta))

def shift_and_rotate(xv,yv,dx,dy,theta):
	return rotate(xv+dx,yv+dy,theta)

def rotate_and_shift(xv,yv,dx,dy,theta):
	(x,y) = rotate(xv,yv,theta)
	return (x+dx,y+dy)
	
def get_rounded_float(values, decimals):
	return map(lambda x: round(x,decimals),values)

def norm(angle):
	if angle >= np.pi:
		angle -= 2*np.pi
	elif angle < -np.pi:
		angle += 2*np.pi
	return(angle)
	
def is_same_point(a,b):
	for v1,v2 in zip(a,b):
		if v1!=v2:
			return False
	return True
	
def degrees_to_radians(deg):
	return deg*np.pi/180
