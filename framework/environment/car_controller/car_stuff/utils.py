import numpy as np
import scipy
		
def rotate(x,y,theta):
	return (x*np.cos(theta)-y*np.sin(theta), x*np.sin(theta)+y*np.cos(theta))

def shift_and_rotate(xv,yv,dx,dy,theta):
	return rotate(xv+dx,yv+dy,theta)

def rotate_and_shift(xv,yv,dx,dy,theta):
	(x,y) = rotate(xv,yv,theta)
	return (x+dx,y+dy)

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

def poly(p, points):
	return points[0] + points[1]*p + (points[2] + points[3]*p)*p**2

def derivative(p, points):
	return points[1] + 2*points[2]*p + 3*points[3]*p**2

def angle(p, U, V):
	Ud = derivative(p,U)
	Vd = derivative(p,V)
	return np.arctan(Vd/Ud) if abs(Ud) > abs(Vd/1000) else (np.pi/2)

def get_poly_length(spline, integration_range): # quad is precise [Clenshaw-Curtis], romberg is generally faster
	U,V = spline
	start, end = integration_range
	return scipy.integrate.romberg(lambda p:np.sqrt(poly(p,U)**2+poly(p,V)**2), start, end, tol=1e-02, rtol=1e-02)
	
def norm(angle):
    if angle >= np.pi:
        angle -= 2*np.pi
    elif angle < -np.pi:
        angle += 2*np.pi
    return angle

def convert_degree_to_radiant(degree):
	return (degree/180)*np.pi
	
def convert_radiant_to_degree(radiant):
	return radiant*(180/np.pi)
	
def get_heading_vector(angle, space=1):
	return (space*np.cos(angle), space*np.sin(angle))
	
def euclidean_distance(a,b):
	return np.sqrt(sum((j-k)**2 for (j,k) in zip(a,b)))
	
def point_is_in_segment(point,segment, epsilon=1e-8):
	a,b = segment
	return np.absolute(euclidean_distance(a,point) + euclidean_distance(b,point) - euclidean_distance(a,b)) <= epsilon

def segments_intersect(AB, CD):
	# Return true if line segments AB and CD intersect
	A, B = AB
	C, D = CD

	# If two segments start/end at same point, consider as *not* intersecting
	for point1 in AB:
		for point2 in CD:
			if point1 == point2:
				return False

	# Explanation here: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
	def ccw(A, B, C):
		Ax, Ay = A
		Bx, By = B
		Cx, Cy = C
		return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
	return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

	
def segment_collide_circle(segment, circle):
	C, radius = circle
	Cx, Cy = C
	A, B = segment
	Ax, Ay = A
	Bx, By = B
	# compute the direction vector D from A to B
	Dx = (Bx-Ax)
	Dy = (By-Ay)
	# Now the line equation is x = Dx*t + Ax, y = Dy*t + Ay with 0 <= t <= 1
	# compute the value t of the closest point in the segment to the circle center (Cx, Cy)
	t = np.clip(Dx*(Cx-Ax) + Dy*(Cy-Ay), 0,1)
	# This is the projection of C on the line from A to B
	# compute the coordinates of the point E on line and closest to C
	E = (t*Dx+Ax,t*Dy+Ay)
	return euclidean_distance(E,C) <= radius # test whether E is in segment and the line intersects the circle or is tangent to circle

def colour_to_hex(colour_name):
	if colour_name == "Grey":
		return "#616A6B"
	elif colour_name == "Olive":
		return "#52BE80"
	elif colour_name == "Brown":
		return "#6E2C00"
	elif colour_name == "Orange":
		return "#D35400"
	elif colour_name == "Purple":
		return "#6C3483"
	elif colour_name == "Red":
		return "#C0392B"
	elif colour_name == "Gold":
		return "#B7950B"
	elif colour_name == "Green":
		return "#196F3D"
	elif colour_name == "Blue":
		return "#2E86C1"
	else:
		return "#000000"


