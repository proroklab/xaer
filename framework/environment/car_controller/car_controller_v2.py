# -*- coding: utf-8 -*-
import numpy as np
from environment.car_controller.car_controller_v1 import CarControllerV1
from environment.car_controller.car_stuff.utils import *

class CarControllerV2(CarControllerV1):
	control_points_per_step = 10

	def get_state_shape(self):
		# There are 2 types of objects (obstacles and lines), each object has 3 numbers (x, y and size)
		# if no obstacles are considered, then there is no need for representing the line size because it is always set to 0
		return [
			(1,self.control_points_per_step,2),
			(1,self.max_obstacle_count,3),
			(self.get_concatenation_size(),)
		]
	
	def __init__(self, config_dict):
		self.max_step = self.max_step_per_spline*self.spline_number
		self.speed_lower_limit = max(self.min_speed_lower_limit,self.min_speed)
		self.meters_per_step = 2*self.max_speed*self.mean_seconds_per_step
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		self.max_steering_noise_angle = convert_degree_to_radiant(self.max_steering_noise_degree)
		# Shapes
		self.control_points_shape = self.get_state_shape()[0]
		self.obstacles_shape = self.get_state_shape()[1]
		self.action_shape = self.get_action_shape()
		
	def get_control_points(self, source_point, source_orientation, source_position): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_goal = self.get_goal(source_position)
		# print(source_position, source_goal)
		control_points = np.zeros((self.control_points_per_step,2), dtype=np.float32)
		source_x, source_y = source_point
		control_distance = (source_goal-source_position)/self.control_points_per_step
		# add control points
		for i in range(self.control_points_per_step):
			cp_x, cp_y = self.get_point_from_position(source_position + (i+1)*control_distance)
			cp_x, cp_y = shift_and_rotate(cp_x, cp_y, -source_x, -source_y, -source_orientation) # get control point with coordinates relative to source point
			control_points[i][0] = cp_x
			control_points[i][1] = cp_y
		return control_points
		
	def get_control_obstacles(self, car_point, car_orientation, obstacles):
		car_x, car_y = car_point
		control_obstacles = []
		for (j, obstacle) in enumerate(obstacles):
			obstacle_point, obstacle_radius = obstacle
			if euclidean_distance(obstacle_point,car_point) <= self.horizon_distance+obstacle_radius:
				ro_x, ro_y = shift_and_rotate(obstacle_point[0], obstacle_point[1], -car_x, -car_y, -car_orientation) # get control point with coordinates relative to car point
				control_obstacles.append((ro_x, ro_y, obstacle_radius))
		if len(control_obstacles) > 0:
			# sort obstacles by euclidean distance from closer to most distant
			control_obstacles.sort(key=lambda t: np.absolute(euclidean_distance((t[0],t[1]),car_point)-t[2]))
		for _ in range(self.max_obstacle_count-len(control_obstacles)):
			control_obstacles.append((0.,0.,0.))
		return control_obstacles
		
	def get_state(self, car_point, car_orientation, car_progress, obstacles):
		control_points_state = np.zeros(self.control_points_shape, dtype=np.float32)
		control_points_state[0] = self.get_control_points(car_point, car_orientation, car_progress)
		obstacles_state = np.zeros(self.obstacles_shape, dtype=np.float32)
		obstacles_state[0] = self.get_control_obstacles(car_point, car_orientation, obstacles)
		return (
			control_points_state, # add control points
			obstacles_state, # add control obstacles
			self.get_concatenation(),
		)
	