# -*- coding: utf-8 -*-
#Copyright (C) 2018 Francesco Sovrano
#
#This file is part of Control Points Generator for Audi Autonomous Driving Cup.
#
#This is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This software is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import optimize
from utils import *
import numpy as np
	
class ControlPointGenerator(object):
	control_points_per_step = 5
	horizon_distance = 1 # meters
	seconds_per_step = 0.1 # seconds
	max_speed = 3 # m/s
	meters_per_step = max_speed*seconds_per_step
	shift_road_origin = 0.3 # meters # positive for keeping right, negative for keeping left, 0 for staying in the middle of the road

	def __init__(self, path):
		self.spline_number = len(path) # need two for properly handling curvers
		assert self.spline_number>0, "Empty path in ControlPointGenerator"
		# setup environment
		self.Us = [s[0:4] for s in path] # Us are x-generator splines
		self.Vs = [s[4:8] for s in path] # Vs are y-generator splines
		self.origins = [s[8:10] for s in path] # origin of each spline
		self.orientations = [s[10] for s in path] # orientation angle of each spline
		self.lengths = [s[11] for s in path] # length of each spline
		self.inversions = [s[12]==1 for s in path]
		
	def get_spline_by_position(self, position):
		return np.clip(int(np.ceil(position)-1),0,self.spline_number-1)
		
	def get_horizon_around(self, position): # get horizon around
		spline = self.get_spline_by_position(position)
		length = self.lengths[spline]
		return self.horizon_distance/length
		
	def get_position_around(self, position): # this function measures the max possible position distance, using: road_length, max_car_speed, seconds_per_step
		spline = self.get_spline_by_position(position)
		length = self.lengths[spline]
		return self.meters_per_step/length
		
	def get_angle_from_position(self, position):
		spline = self.get_spline_by_position(position)
		return angle(position-spline, self.Us[spline], self.Vs[spline])+self.orientations[spline]

	def get_point_from_position(self, position):
		spline = self.get_spline_by_position(position)
		inversion = self.inversions[spline]
		origin = self.origins[spline]
		orientation = self.orientations[spline]
		U = self.Us[spline]
		V = self.Vs[spline]
		relative_position = position-spline
		if inversion:
			relative_position = 1 - relative_position
		x,y = rotate_and_shift(poly(relative_position,U), poly(relative_position,V), origin[0], origin[1], orientation)
		if self.shift_road_origin == 0:
			return x, y
		# else apply shift to road origin
		arctan = self.get_angle_from_position(position) # arctangent
		x,y = shift_and_rotate(x, y, -origin[0], -origin[1], -arctan) # point relative to origin
		shift = self.shift_road_origin * (1 if inversion else -1)
		return rotate_and_shift(x, y+shift, origin[0], origin[1], arctan)
		
	def find_closest_spline_position(self, point, start_position=0, end_position=1):
		return optimize.minimize_scalar(lambda pos: euclidean_distance(point, self.get_point_from_position(pos)), method='bounded', bounds=(start_position,end_position)).x
			
	def find_closest_position(self, point, previous_position=None): # Find the closest spline
		if previous_position is not None:
			position_around = max(self.get_position_around(previous_position),1) # usually 1 is big enough, but sometimes the road length may be too small
			# N.B. properly tuning position_around may significantly speed up the search for the closest_spline_position
			return self.find_closest_spline_position(point=point, start_position=previous_position, end_position=previous_position+position_around)
		elif self.spline_number == 1:
			return self.find_closest_spline_position(point=point)
		else:
			# road is non-convex function, thus we have to find separately closest position in all splines
			closest_position_generator = (self.find_closest_spline_position(point=point, start_position=spline, end_position=spline+1) for spline in range(self.spline_number))
			position_distance_generator = [(pos, euclidean_distance(point,self.get_point_from_position(pos))) for pos in closest_position_generator]
			return min(position_distance_generator, key=lambda t: t[1])[0]

	def find_closest_point(self, point, previous_position=None):
		closest_position = self.find_closest_position(point, previous_position)
		return self.get_point_from_position(closest_position)

	def get_control_points(self, source_point, source_orientation, source_position, source_speed): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_goal = source_position + self.get_horizon_around(source_position)
		# print(source_position, source_goal)
		control_points = np.zeros((self.control_points_per_step,4), dtype=np.float32)
		source_x, source_y = source_point
		control_distance = (source_goal-source_position)/self.control_points_per_step
		# add control points
		for i in range(self.control_points_per_step):
			pos = source_position + (i+1)*control_distance
			cp_x, cp_y = self.get_point_from_position(pos)
			x,y = shift_and_rotate(cp_x, cp_y, -source_x, -source_y, -source_orientation) # get control point with coordinates relative to source point
			steering_angle, new_orientation = self.get_steering_angle((x,y), source_speed, source_orientation, self.get_angle_from_position(pos))
			control_points[i] = [x,y,steering_angle,new_orientation]
		return control_points

	def get_steering_angle(self, control_point, speed, current_orientation, tangent_angle):
		gamma = 40
		wheelbase = 2
		xc,yc = control_point
		compensation = np.arctan((yc-2)/(speed*gamma))
		#if comp > .2: ignore_comp = True
		new_orientation = norm(tangent_angle + compensation)
		steering_angle = np.arctan((new_orientation-current_orientation) * wheelbase / speed)
		return steering_angle, new_orientation
