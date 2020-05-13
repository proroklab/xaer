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

from path_generator import PathGenerator
from control_point_generator import ControlPointGenerator

def init(map_file, maneuvers_file, sign_file):
	global path_gen
	path_gen = PathGenerator(map_file, maneuvers_file, sign_file) # global variable, load this module once

def get_closest_road_spline(point): # 2D point
	global path_gen
	road = path_gen.find_closest_road(point)
	return path_gen.get_road_spline(road) # U,V, origin, orientation

def get_path(point, direction='forward'): # 2D point
	global path_gen
	return path_gen.get_path(source_point=point, direction=direction) # [(U,V, origin, orientation),...]
	
def get_path_by_action(action, extra_action, point, direction='forward'):
	global path_gen, old_road, road # keep track of old action ending points, in order to properly select direction
	try: # Determine if variable is defined
		old_road, road
	except NameError:
		old_road = None
		road = path_gen.find_closest_road(point)
	action_path, old_road, road = path_gen.get_path_piece(action=action, extra_action=extra_action, old_road=old_road, road=road, direction=direction)
	return action_path
	
def setup_splines_for_control_point_generation(path): # path is a list of splines
	global cpg_gen
	cpg_gen = ControlPointGenerator(path)
	
def get_path_progress(point, progress):
	global cpg_gen
	return cpg_gen.find_closest_position(point, progress if progress >= 0 else None)
	
def get_normalized_path_progress(point, progress):
	global cpg_gen
	progress = cpg_gen.find_closest_position(point, progress if progress >= 0 else None)
	return progress/cpg_gen.spline_number # in [0,1]
	
def get_closest_road_point(point, progress):
	global cpg_gen
	return cpg_gen.find_closest_point(point, progress if progress >= 0 else None)
	
def get_control_points(point, angle, spline_position, speed):
	global cpg_gen
	return cpg_gen.get_control_points(point, angle, spline_position, speed).tolist()
