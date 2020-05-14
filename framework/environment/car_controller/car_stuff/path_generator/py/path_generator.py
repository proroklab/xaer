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

from utils import *
from control_point_generator import ControlPointGenerator
from lxml import etree as et
import math
from collections import deque
	
class PathGenerator(object):

	def __init__(self, map_file, maneuvers_file, sign_file):
		# roads
		map_tree = et.parse(map_file)
		self.roads = {self.get_id(road):road for road in map_tree.xpath("//road")}
		self.road_splines = {k:self.get_road_spline(v) for k,v in self.roads.items()}
		self.junctions = {self.get_id(junction):junction for junction in map_tree.xpath("//junction")}
		self.connections = {int(connection.get("connectingRoad")):{"road":connection,"junction_id":id} for id,junction in self.junctions.items() for connection in junction.xpath("connection")}
		self.incoming_junction_roads = {int(connection["road"].get("incomingRoad")):connection["junction_id"] for connection in self.connections.values()}
		# maneuvers
		self.sectors = et.parse(maneuvers_file).xpath("//AADC-Sector")
		self.actions = [maneuver.get("action") for sector in self.sectors for maneuver in sector.xpath("AADC-Maneuver")]
		# parking spaces
		signs_tree = et.parse(sign_file)
		self.parking_spaces = {
			self.get_id(space):{ # id:(space,road)
				"space":space,
				"road":self.find_closest_road(self.get_parking_space_point(space))
			}
			for space in signs_tree.xpath("//parkingSpace")
		}
		#parking_space = self.parking_spaces[park_id]["space"]
		#parking_space_point = self.get_parking_space_point(parking_space)
		#x,y = parking_space_point 

	def find_closest_road(self, point):
		# print(point)
		road_distances = [(road_id, euclidean_distance(point, ControlPointGenerator(path=[spline]).find_closest_point(point))) for road_id,spline in self.road_splines.items()]
		closest_road_id = min(road_distances, key=lambda t: t[1])[0] # road, distance
		return self.roads[closest_road_id]

	def get_road_spline(self, road):
		geometry = road.xpath("planView/geometry")[0]
		poly = geometry.xpath("paramPoly3")[0]
		spline = (
			poly.get("aU"), poly.get("bU"), poly.get("cU"), poly.get("dU"), # U
			poly.get("aV"), poly.get("bV"), poly.get("cV"), poly.get("dV"), # V
			geometry.get("x"), geometry.get("y"), # origin
			geometry.get("hdg"), # orientation
			geometry.get("length"), # length
			0 # starting pos
		)
		# format output
		return list(map(float,spline))
		
	def get_parking_space_point(self, parking_space):
		point = (parking_space.get("x"), parking_space.get("y"))
		return list(map(float,point))

	def get_id(self, node):
		return int(node.get("id"))
		
	def find_road_by_id(self, id):
		return self.roads[int(id)]

	def get_successor_info(self, road):
		successor = road.xpath("link/successor")[0]
		id = int(successor.get("elementId"))
		type = successor.get("elementType")
		return id, type

	def get_predecessor_info(self, road):
		predecessor = road.xpath("link/predecessor")[0]
		id = int(predecessor.get("elementId"))
		type = predecessor.get("elementType")
		return id, type

	def get_next_road_id(self, road, old_road=None, direction='forward'):
		current_id = self.get_id(road)
		predecessor_id, predecessor_type = self.get_predecessor_info(road)
		successor_id, successor_type = self.get_successor_info(road)
		if old_road is not None:
			old_road_id = self.get_id(old_road)
			old_junction_id = None
			if old_road_id in self.incoming_junction_roads:
				old_junction_id = self.incoming_junction_roads[old_road_id]
			if old_road_id in self.connections:
				old_junction_id = self.connections[old_road_id]["junction_id"]
			next_id, next_type = self.get_successor_info(road) if successor_id != old_junction_id and successor_id != old_road_id else self.get_predecessor_info(road)
		else: # first road of path, direction variable selects the direction: forward (successor is up/right), backward (successor is down/left)
			next_id, next_type = (predecessor_id, predecessor_type) if self.next_is_predecessor(road, direction) else (successor_id, successor_type)
		# print("current_id",current_id,"predecessor_id",predecessor_id,"successor_id",successor_id,"next_id",next_id,"current_point",self.get_road_point(road))
		return next_id, next_type
		
	def next_is_predecessor(self, road, direction):
		predecessor_id, predecessor_type = self.get_predecessor_info(road)
		successor_id, successor_type = self.get_successor_info(road)
		road_point = self.get_road_point(road)
		predecessor_point = self.get_road_point(self.find_road_by_id(predecessor_id))
		successor_point = self.get_road_point(self.find_road_by_id(successor_id))
		delta_x = predecessor_point[0]-road_point[0] if predecessor_type == "road" else -(successor_point[0]-road_point[0])
		if delta_x != 0: # moving up/down
			pick_predecessor = delta_x > 0 if direction == 'forward' else delta_x < 0
		else: # moving left/right
			delta_y = predecessor_point[1]-road_point[1] if predecessor_type == "road" else -(successor_point[1]-road_point[1])
			pick_predecessor = delta_y > 0 if direction == 'forward' else delta_y < 0
		return pick_predecessor
		
	def get_connections(self, junction_id, incoming_road_id):
		connections = self.junctions[int(junction_id)].xpath("connection[@incomingRoad = '{}']".format(incoming_road_id))
		return [c.get("connectingRoad") for c in connections]
		
	def get_outcoming_junction_roads(self, road, junction_id):
		current_id = self.get_id(road)
		connection_ids = self.get_connections(junction_id=junction_id, incoming_road_id=current_id)
		# print("junction_id",junction_id,"incoming_road_id",current_id,"connections",connection_ids)
		connections = [self.find_road_by_id(id) for id in connection_ids]
		outcoming_roads = [(self.get_successor_info(c)[0],c) for c in connections] # tuple: (next road, connection)
		outcoming_roads.extend([(self.get_predecessor_info(c)[0],c) for c in connections])
		return [(self.find_road_by_id(outcoming_road_id),connection) for outcoming_road_id, connection in outcoming_roads if outcoming_road_id != current_id] # filter out current road
		
	def get_next_road(self, road, old_road=None, direction='forward'):
		current_id = self.get_id(road)
		return self.get_next_road_id(road,old_road,direction)

	def get_road_path(self, source_point, direction='forward'):
		old_road = None
		road = self.find_closest_road(source_point)
		path = []
		for action in self.actions:
			action_path, old_road, road = self.get_action_roads(action=action[0], extra_action=action[1], old_road=old_road, road=road, direction=direction)
			path.extend(action_path)
		return path
		
	def get_action_roads(self, action, extra_action, old_road, road, direction):
		path = [road]
		next_id, next_type = self.get_next_road(road=road, old_road=old_road, direction=direction)
		park_road_id = self.get_id(self.parking_spaces[extra_action]["road"]) if "park" in action else None
	# Facing straight road
		while next_type == "road": # keep adding roads until you find a junction # may diverge in map without junctions
			# update next road info
			old_road = road
			road = self.find_road_by_id(next_id)
			next_id, next_type = self.get_next_road(road=road, old_road=old_road, direction=direction)
			# add next road to path
			path.append(road)
			if next_id == park_road_id:
				return path, old_road, road
	# Facing junction: we have to decide where to go
		# next_roads now contains the connections (not the next roads!) but we need next roads to measure their angle with respect to current road
		# we get next roads from connections
		outcoming_junction_roads = self.get_outcoming_junction_roads(road, next_id)
		# get road angles to know whether to turn left/right/straight
		# sort road angles from smallest to biggest
		road_angles = sorted([(outcoming_road,self.get_road_angle(road,outcoming_road),connection) for outcoming_road, connection in outcoming_junction_roads], key=lambda tup: tup[1])
		# print("road angles (next road, angle, connection)", [(self.get_id(r[0]), r[1], self.get_id(r[2])) for r in road_angles]) # for debugging
		# find the id of the desired road
		left_id = -1 # biggest road (angle) is on left
		right_id = 0 # smallest road (angle) is on right
		# print("going {}".format(action))
		if 'right' in action:
			action_id = right_id
		elif 'left' in action:
			action_id = left_id
		else:
			if len(road_angles) > 2: # if there are more than 3 roads, you may need to specify a road selection policy -> this code does not support road selection policies yet
				action_id = len(road_angles)//2 # pick the road in the middle
			else: # only two roads in junction
				action_id = right_id if road_angles[left_id][1] < np.pi/3 else left_id
		# add connection and next road to path
		path.append(road_angles[action_id][2])
		# update next road info
		old_road = road
		road = road_angles[action_id][0]
		return path, old_road, road
	
	def get_road_splines(self, path, direction):
		splines = deque()
		last_spline = self.get_road_spline(path[-1])
		splines.appendleft(last_spline) # add on top
		last_origin = get_rounded_float(last_spline[8:10],2) # round to avoid numerical errors when comparing points!
		for i in range(-2,-len(path)-1,-1): # from bottom to top
			spline = self.get_road_spline(path[i])
			origin = get_rounded_float(spline[8:10],2) # round to avoid numerical errors when comparing points!
			end = get_rounded_float(ControlPointGenerator(path=[spline]).get_point_from_position(1),2) 
			# print(map(lambda x: round(x,2),ControlPointGenerator(path=[spline]).get_point_from_position(0)), map(lambda x: round(x,2),ControlPointGenerator(path=[spline]).get_point_from_position(1)))
			if not is_same_point(end,last_origin): # move on this spline from 1 to 0 instead of 0 to 1
				# print(end,origin,last_origin,np.remainder(math.atan2(end[1]-origin[1],end[0]-origin[0]),2*np.pi)*180/np.pi)
				last_origin = end
				spline[-1] = 1. # change starting point to 1, default is 0
			else:
				last_origin = origin
			splines.appendleft(spline) # add on top
		if self.next_is_predecessor(path[-1], direction): # invert all starting points
			for s in splines:
				s[-1] = 1.-s[-1] # invert starting point
		return list(splines)
	
	def get_path(self, source_point, direction='forward'):
		path = self.get_road_path(source_point, direction)
		return self.get_road_splines(path, direction)
		
	def get_path_piece(self, action, extra_action, old_road, road, direction='forward'):
		action_path, old_road, road = self.get_action_roads(action, extra_action, old_road, road, direction)
		return self.get_road_splines(action_path, direction), old_road, road

	def get_road_point(self, road):
		geometry = road.xpath("planView/geometry")[0]
		road_point = (geometry.get("x"), geometry.get("y"))
		# format output
		return tuple(map(float,road_point))

	def get_road_angle(self, start_road, end_road): # in radians
		x1,y1 = self.get_road_point(start_road)
		x2,y2 = self.get_road_point(end_road)
		dy = y2-y1
		dx = x2-x1
		angle = math.atan2(dy,dx) # in [-pi,pi]
		# print("start_road",self.get_id(start_road),"end_road",self.get_id(end_road),"delta y",y2-y1,"delta x",x2-x1,"ratio",(y2-y1)/(x2-x1),"angle",angle)
		if dy > 0 and dx < 0:
			return np.pi-angle
		if dy < 0 and dx > 0:
			return np.pi+angle
		if dy < 0 and dx < 0:
			return -angle-np.pi/2
		if dy > 0 and dx > 0:
			return angle+np.pi/2
			
	def build_spline_from_boundary_points(self, start_point, end_point):
		x0, y0 = start_point
		x1, y1 = end_point
		dy = y1-y0
		dx = x1-x0
		angle = math.atan2(dy,dx) # in [-pi,pi]
		length = math.sqrt(dx**2 + dy**2)
		return (x0, x1-x0, 0, 0, # U
			y0, y1-y0, 0, 0, # V
			x0, y0, # origin
			angle, # orientation
			length, # length
			0) # start position
			
	def get_parking_splines(self, point, degrees):
		points = [(0,0),(0,1),(0.5,1.5),(1,2)]
		return [self.build_spline_from_boundary_points(points[i],points[i+1]) for i in range(len(points)-1)]
