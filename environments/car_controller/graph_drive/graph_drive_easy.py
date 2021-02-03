# -*- coding: utf-8 -*-
from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from environments.car_controller.utils import *
from environments.car_controller.graph_drive.lib.roads import *
from environments.car_controller.grid_drive.lib.road_cultures import EasyRoadCulture
import random
import gym

class GraphDriveEasy(gym.Env):
	mean_seconds_per_step = 0.1 # in average, a step every n seconds
	horizon_distance = 3 # meters
	track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	wheelbase = 0.9 # meters # https://en.wikipedia.org/wiki/Wheelbase
	# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
	min_speed = 0.4 # m/s
	max_speed = 1.6 # m/s
	# the fastest car has max_acceleration 9.25 m/s^2 (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
	# the slowest car has max_acceleration 0.7 m/s^2 (http://automdb.com/max_acceleration)
	max_acceleration = 0.7 # m/s^2
	# the best car has max_deceleration 29.43 m/s^2 (https://www.quora.com/What-can-be-the-maximum-deceleration-during-braking-a-car?share=1)
	# a normal car has max_deceleration 7.1 m/s^2 (http://www.batesville.k12.in.us/Physics/PhyNet/Mechanics/Kinematics/BrakingDistData.html)
	max_deceleration = 7.1 # m/s^2
	max_steering_degree = 30
	max_step = 1000
	max_distance_to_path = 0.3 # meters
	min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	max_speed_noise = 0.25 # m/s
	max_steering_noise_degree = 2
	# multi-road related stuff
	junction_number = 32
	max_dimension = 64
	map_size = (max_dimension, max_dimension)
	max_road_length = max_dimension*2/3
	CULTURE = EasyRoadCulture

	def get_state_shape(self):
		return [
			{  # Beginning and end of closest road to the agent (the one it's driving on)
				'low': -15,
				'high': 15,
				'shape': (1, 2, 2 + self.obs_road_features), # current road view: relative coordinates of road.start.pos and road.end.pos
			},
			{  # Junctions
				'low': -15,
				'high': 15,
				'shape': ( # closest junctions view
					2, # number of junctions close to current road
					Junction.max_roads_connected, 
					1 + self.obs_road_features,  # relative heading vector + road features
				),
			},
			{  # Agent features
				'low': -1,
				'high': self.max_speed/self.speed_lower_limit,
				'shape': (self.get_concatenation_size() + self.obs_car_features,)
			},
		]

	def get_state(self, car_point, car_orientation):
		return (
			*self.get_view(car_point, car_orientation), 
			np.concatenate([self.get_concatenation(), self.road_network.agent.binary_features()], axis=-1),

		)

	def get_concatenation_size(self):
		return 3
		
	def get_concatenation(self):
		return np.array((
			self.steering_angle/self.max_steering_angle, 
			self.speed/self.max_speed, 
			self.speed/self.speed_upper_limit,
		), dtype=np.float32)

	def get_reward(self, car_speed, car_point, old_car_point): # to finish
		def terminal_reward(is_positive,label):
			return (1 if is_positive else -1, True, label) # terminate episode
		def step_reward(is_positive,label):
			space_traveled = car_speed*self.seconds_per_step # space traveled
			return (space_traveled if is_positive else -space_traveled, False, label) # terminate episode
		
		distance_from_junction = min(euclidean_distance(self.closest_junctions[0].pos, car_point), euclidean_distance(self.closest_junctions[1].pos, car_point))
		# print(distance_from_junction, self.junction_around)
		if distance_from_junction > self.junction_around:
			# "Valid colour" rule
			# "Stay on the road" rule
			if self.distance_to_closest_road > 2*self.max_distance_to_path: 
				return terminal_reward(is_positive=False, label='stay_on_the_road')
			# "Follow the lane" rule # this has an higher priority than the 'respect_speed_limit' rule
			if self.distance_to_closest_road > self.max_distance_to_path:
				return step_reward(is_positive=False, label='follow_lane')
		# "Respect the speed limit" rule
		if car_speed > self.speed_upper_limit:
			return step_reward(is_positive=False, label='respect_speed_limit')
		return step_reward(is_positive=True, label='move_forward')

	def __init__(self):
		self.speed_lower_limit = max(self.min_speed_lower_limit,self.min_speed)
		self.meters_per_step = 2*self.max_speed*self.mean_seconds_per_step
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		self.max_steering_noise_angle = convert_degree_to_radiant(self.max_steering_noise_degree)
		self.road_network = RoadNetwork(self.CULTURE, map_size=self.map_size, max_road_length=self.max_road_length)
		self.junction_around = min(self.max_distance_to_path*2, self.road_network.min_junction_distance/8)

		self.culture = self.CULTURE(road_options={
			'motorway': 1/2,
			'stop_sign': 1/2,
			'school': 1/2,
			'single_lane': 1/2,
			'town_road': 1/2,
			'roadworks': 1/8,
			'accident': 1/8,
			'heavy_rain': 1/2,
			'congestion_charge': 1/8,
		}, agent_options={
			'emergency_vehicle': 1/5,
			'heavy_vehicle': 1/4,
			'worker_vehicle': 1/3,
			'tasked': 1/2,
			'paid_charge': 1 / 2,
			'speed': self.road_network.normalise_speed(self.min_speed, self.max_speed, self.max_speed),
		})
		self.obs_road_features = len(self.culture.properties)  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(
			self.culture.agent_properties) - 1  # Number of binary CAR features in Hard Culture (excluded speed)
		# # Spaces
		# self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # steering angle, continuous control without softmax
		# shapes = [gym.spaces.Box(**shape, dtype=np.float32) for shape in self.get_state_shape()]
		# features = gym.spaces.Dict({
		# 	"cnn": gym.spaces.Dict({
		# 		"grid": gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION, self.obs_road_features+2]), # Features representing the grid + visited cells + current position
		# 	}),
		# 	"fc": gym.spaces.Dict(fc_dict),
		# })
		# self.observation_space = gym.spaces.Tuple(shapes, )
		# Spaces
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,),
										   dtype=np.float32)  # steering angle, continuous control without softmax
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.Box(**shape, dtype=np.float32)
			for shape in self.get_state_shape()
		])

	@staticmethod
	def normalize_point(p):
		return (p[0]/GraphDriveEasy.map_size[0], p[1]/GraphDriveEasy.map_size[1])

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		# get road view
		source_x, source_y = source_point
		j1, j2 = self.closest_junctions
		# Get road view
		road_view = ( # 4x2
			j1.pos,
			j2.pos,
			*self.closest_road.binary_features(),
		)
		road_view = map(lambda x: shift_and_rotate(*x, -source_x, -source_y, -source_orientation), road_view)
		road_view = map(self.normalize_point, road_view) # in [-1,1]
		road_view = tuple(road_view)
		road_view = np.array((road_view,), dtype=np.float32)
		# Get junction view
		junction_view = np.array([ # 2 x Junction.max_roads_connected x (1+1)
			[
				(
					(road.get_orientation_relative_to(source_orientation) % two_pi)/two_pi, # in [0,1]
					*road.binary_features(),
					# RoadNetwork.all_road_colours.index(road.colour)/len(RoadNetwork.all_road_colours), # in [0,1]
				)
				for road in j.roads_connected
			] + [(-1,-1)]*(Junction.max_roads_connected-len(j.roads_connected))
			for j in (j1,j2)
		], dtype=np.float32)
		return road_view, junction_view
	
	def reset(self):
		self.is_over = False
		self.episode_statistics = {}
		self._step = 0
		###########################
		self.seconds_per_step = self.get_step_seconds()
		# self.car_colour = random.choice(RoadNetwork.all_road_colours)
		# self.normalised_car_colour = RoadNetwork.all_road_colours.index(self.car_colour)/len(RoadNetwork.all_road_colours)
		# self.car_feasible_colours = self.road_network.feasible_road_colours(self.car_colour)

		# car position
		self.car_point = self.road_network.set(self.junction_number)
		self.car_orientation = (2*np.random.random()-1)*np.pi # in [-pi,pi]
		self.distance_to_closest_road, self.closest_road, self.closest_junctions = self.road_network.get_closest_road_and_junctions(self.car_point)
		# speed limit
		self.speed_upper_limit = self.speed_lower_limit + (self.max_speed-self.speed_lower_limit)*np.random.random() # in [speed_lower_limit,max_speed]
		# steering angle & speed
		self.speed = self.min_speed + (self.max_speed-self.min_speed)*np.random.random() # in [min_speed,max_speed]
		self.steering_angle = 0
		# init concat variables
		self.last_reward = 0
		self.last_reward_type = 'move_forward'
		self.last_action_mask = None
		self.last_state = self.get_state(car_point=self.car_point, car_orientation=self.car_orientation)
		# init log variables
		self.cumulative_reward = 0
		self.avg_speed_per_steps = 0
		return self.last_state

	def move(self, point, orientation, steering_angle, speed, add_noise=False):
		# https://towardsdatascience.com/how-self-driving-cars-steer-c8e4b5b55d7f?gi=90391432aad7
		# Add noise
		if add_noise:
			steering_angle += (2*np.random.random()-1)*self.max_steering_noise_angle
			steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle) # |steering_angle| <= max_steering_angle, ALWAYS
			speed += (2*np.random.random()-1)*self.max_speed_noise
		# Get new angle
		# https://www.me.utexas.edu/~longoria/CyVS/notes/07_turning_steering/07_Turning_Kinematically.pdf
		angular_velocity = speed*np.tan(steering_angle)/self.wheelbase
		# get normalized new orientation
		new_orientation = np.mod(orientation + angular_velocity*self.seconds_per_step, 2*np.pi) # in [0,2*pi)
		# Move point
		x, y = point
		dir_x, dir_y = get_heading_vector(angle=new_orientation, space=speed*self.seconds_per_step)
		return (x+dir_x, y+dir_y), new_orientation

	def get_steering_angle_from_action(self, action): # action is in [-1,1]
		return action*self.max_steering_angle # in [-max_steering_angle, max_steering_angle]
		
	def get_acceleration_from_action(self, action): # action is in [-1,1]
		return action*(self.max_acceleration if action >= 0 else self.max_deceleration) # in [-max_deceleration, max_acceleration]
		
	def accelerate(self, speed, acceleration):
		# use seconds_per_step instead of mean_seconds_per_step, because this way the algorithm is able to explore more states and train better
		# return np.clip(speed + acceleration*self.mean_seconds_per_step, self.min_speed, self.max_speed)
		return np.clip(speed + acceleration*self.seconds_per_step, self.min_speed, self.max_speed)
		
	def get_step_seconds(self):
		return np.random.exponential(scale=self.mean_seconds_per_step)

	def step(self, action_vector):
		# first of all, get the seconds passed from last step
		self.seconds_per_step = self.get_step_seconds()
		# compute new steering angle
		self.steering_angle = self.get_steering_angle_from_action(action=action_vector[0])
		# compute new acceleration
		self.acceleration = self.get_acceleration_from_action(action=action_vector[1])
		# compute new speed
		self.speed = self.accelerate(speed=self.speed, acceleration=self.acceleration)
		# move car
		old_car_point = self.car_point
		self.car_point, self.car_orientation = self.move(
			point=self.car_point, 
			orientation=self.car_orientation, 
			steering_angle=self.steering_angle, 
			speed=self.speed, 
			add_noise=True
		)
		self.distance_to_closest_road, self.closest_road, self.closest_junctions = self.road_network.get_closest_road_and_junctions(self.car_point, self.closest_junctions)
		# compute perceived reward
		reward, dead, reward_type = self.get_reward(
			car_speed=self.speed, 
			car_point=self.car_point, 
			old_car_point=old_car_point, 
		)
		# compute new state (after updating progress)
		state = self.get_state(
			car_point=self.car_point, 
			car_orientation=self.car_orientation,
		)
		# update last action/state/reward
		self.last_state = state
		self.last_reward = reward
		self.last_reward_type = reward_type
		# update cumulative reward
		self.cumulative_reward += reward
		self.avg_speed_per_steps += self.speed
		# update step
		self._step += 1
		completed_track = self.last_reward_type == 'goal'
		out_of_time = self._step >= self.max_step
		terminal = dead or completed_track or out_of_time
		if terminal: # populate statistics
			self.is_over = True
			stats = {
				"avg_speed": self.avg_speed_per_steps/self._step,
				"completed_track": 1 if completed_track else 0,
				"out_of_time": 1 if out_of_time else 0,
			}
			self.episode_statistics = stats
		return [state, reward, terminal, {'explanation':reward_type}]
		
	
	def get_info(self):
		return f"speed={self.speed}, steering_angle={self.steering_angle}, orientation={self.car_orientation}\n"
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5))
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [Circle(junction.pos,0.2,color='r') for junction in self.road_network.junctions]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)

		# [Car]
		car_x, car_y = self.car_point
		car_handle = ax.scatter(car_x, car_y, marker='o', color='g', label='Car')
		# [Heading Vector]
		dir_x, dir_y = get_heading_vector(angle=self.car_orientation)
		heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')

		# [Roads]
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			# print("Drawing road {} {}".format(road[0], road[1]))
			path_handle, = ax.plot(road_pos[0], road_pos[1], color=colour_to_hex(road.colour), lw=2, alpha=0.5, label='Roads')

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car_handle]
		ax.legend(handles=handles)
		# Draw plot
		figure.suptitle('[Angle]{1:.2f}° [Orientation]{4:.2f}° \n [Speed]{0:.2f} m/s [Limit]{3:.2f} m/s [Step]{2}'.format(self.speed, convert_radiant_to_degree(self.steering_angle), self._step, self.speed_upper_limit, convert_radiant_to_degree(self.car_orientation)))
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		figure.clear()
		return {'RGB': data} # RGB array
				
	def get_statistics(self):
		return self.episode_statistics
