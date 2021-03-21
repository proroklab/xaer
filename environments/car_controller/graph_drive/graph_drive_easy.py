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
from gym.utils import seeding

class GraphDriveEasy(gym.Env):
	random_seconds_per_step = False # whether to sample seconds_per_step from an exponential distribution
	mean_seconds_per_step = 0.5 # in average, a step every n seconds
	# track = 0.4 # meters # https://en.wikipedia.org/wiki/Axle_track
	wheelbase = 0.45 # meters # https://en.wikipedia.org/wiki/Wheelbase
	# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
	min_speed = 0.2 # m/s
	max_speed = 1.2 # m/s
	# the fastest car has max_acceleration 9.25 m/s^2 (https://en.wikipedia.org/wiki/List_of_fastest_production_cars_by_acceleration)
	# the slowest car has max_acceleration 0.7 m/s^2 (http://automdb.com/max_acceleration)
	max_acceleration = 1 # m/s^2
	# the best car has max_deceleration 29.43 m/s^2 (https://www.quora.com/What-can-be-the-maximum-deceleration-during-braking-a-car?share=1)
	# a normal car has max_deceleration 7.1 m/s^2 (http://www.batesville.k12.in.us/Physics/PhyNet/Mechanics/Kinematics/BrakingDistData.html)
	max_deceleration = 7 # m/s^2
	max_steering_degree = 45
	max_step = 200
	max_distance_to_path = 1 # meters
	# min_speed_lower_limit = 0.7 # m/s # used together with max_speed to get the random speed upper limit
	# max_speed_noise = 0.25 # m/s
	# max_steering_noise_degree = 2
	max_speed_noise = 0 # m/s
	max_steering_noise_degree = 0
	# multi-road related stuff
	max_dimension = 16
	map_size = (max_dimension, max_dimension)
	junction_number = 16
	max_roads_per_junction = 4
	junction_radius = 1
	min_junction_distance = 2.5*junction_radius
	CULTURE = EasyRoadCulture
	MAX_NORMALISED_SPEED = 120

	def get_state_shape(self):
		return [
			{  # Closest road to the agent (the one it's driving on), sorted by relative position
				'low': -1,
				'high': 1,
				'shape': (
					2 + 2 + self.obs_road_features + 1, # road properties: road.start.pos + road.end.pos + road.af_features + road.is_new_road
				),
			},
			{  # Roads directly connected to the closest road to the agent (the one it's driving on), sorted by relative position
				'low': -1,
				'high': 1,
				'shape': ( # closest junctions view
					2, # junctions attached to the current road
					self.max_roads_per_junction, # maximum number of roads per junction
					2 + 2 + self.obs_road_features + 1,  # road properties: road.start.pos + road.end.pos + road.af_features + road.is_new_road
				),
			},
			{  # Agent features
				'low': -1,
				'high': 1,
				'shape': (self.get_agent_state_size() + self.obs_car_features,),
			},
		]

	def get_state(self, car_point, car_orientation):
		return (
			*self.get_view(car_point, car_orientation), 
			np.concatenate([
				self.get_agent_state(),
				self.road_network.agent.binary_features(), 
			], axis=-1),
		)

	def get_agent_state_size(self):
		return 4 # normalised steering angle + normalised speed
		
	def get_agent_state(self):
		return np.array((
			self.steering_angle/self.max_steering_angle, # normalised steering angle
			self.speed/self.max_speed, # normalised speed
			min(1, self.distance_to_closest_road/self.max_distance_to_path),
			self.is_in_junction(self.car_point),
		), dtype=np.float32)

	def get_reward(self, car_speed, car_point, old_car_point, visiting_new_road): # to finish
		def terminal_reward(is_positive,label):
			return (1 if is_positive else -1, True, label) # terminate episode
		def non_terminal_reward(is_positive,label):
			return (1 if is_positive else -1, False, label) # do not terminate episode
		# def step_reward(is_positive,label):
		# 	normalised_space_traveled = (car_speed - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
		# 	return (normalised_space_traveled if is_positive else -normalised_space_traveled, False, label) # do not terminate episode
		def null_reward(label):
			return (0, False, label) # do not terminate episode

		#######################################
		# "Is in junction" rule
		if self.is_in_junction(car_point):
			return null_reward(label='is_in_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return terminal_reward(is_positive=False, label='not_staying_on_the_road')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		self.road_network.agent.assign_property_value("Speed", self.road_network.normalise_speed(self.min_speed, self.max_speed, car_speed))
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		explanation_list_with_label = lambda l: list(map(lambda x:(l,x), explanation_list)) if explanation_list else l
		if not following_regulation:
			return terminal_reward(is_positive=False, label=explanation_list_with_label('not_following_regulation'))
		#######################################
		# "Visit new roads" rule
		if self.closest_road.is_visited: # visiting a previously seen reward gives no bonus
			return null_reward(label=explanation_list_with_label('not_visiting_new_roads'))
		#######################################
		# "Explore new roads" rule
		if visiting_new_road: # visiting a new road for the first time is equivalent to get a bonus reward
			return non_terminal_reward(is_positive=True, label=explanation_list_with_label('exploring_a_new_road'))
		#######################################
		# "Move forward" rule
		return null_reward(label=explanation_list_with_label('moving_forward'))

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		random.seed(seed)
		return [seed]

	def __init__(self, config):
		self.seed(config.worker_index * config.num_workers)
		self.viewer = None
		self.max_steering_angle = convert_degree_to_radiant(self.max_steering_degree)
		self.max_steering_noise_angle = convert_degree_to_radiant(self.max_steering_noise_degree)

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
			'paid_charge': 1/2,
			'speed': self.MAX_NORMALISED_SPEED,
		})
		self.road_network = RoadNetwork(
			self.culture, 
			map_size=self.map_size, 
			min_junction_distance=self.min_junction_distance,
			max_roads_per_junction=self.max_roads_per_junction,
		)
		self.obs_road_features = len(self.culture.properties)  # Number of binary ROAD features in Hard Culture
		self.obs_car_features = len(self.culture.agent_properties) - 1  # Number of binary CAR features in Hard Culture (excluded speed)
		# Spaces
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # steering angle and speed
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.Box(**shape, dtype=np.float32)
			for shape in self.get_state_shape()
		])

	@staticmethod
	def normalize_point(p):
		return (np.clip(p[0]/GraphDriveEasy.map_size[0],-1,1), np.clip(p[1]/GraphDriveEasy.map_size[1],-1,1))

	def get_view(self, source_point, source_orientation): # source_orientation is in radians, source_point is in meters, source_position is quantity of past splines
		source_x, source_y = source_point
		shift_rotate_normalise_point = lambda x: self.normalize_point(shift_and_rotate(*x, -source_x, -source_y, -source_orientation))
		j1, j2 = self.closest_junctions
		# Get road view
		road_points = ( # 2x2
			j1.pos,
			j2.pos,
		)
		relative_road_points = tuple(map(shift_rotate_normalise_point, road_points))
		road_view = sum(sorted(relative_road_points),()) + self.closest_road.binary_features(as_tuple=True) + (1 if self.closest_road.is_visited else 0,)
		road_view = np.array(road_view, dtype=np.float32)
		# Get junction view
		junction_view = np.array([ # 2 x self.max_roads_per_junction x (1+1)
			sorted([
				(
					*shift_rotate_normalise_point(road.start.pos),
					*shift_rotate_normalise_point(road.end.pos),
					*road.binary_features(as_tuple=True), # in [0,1]
					1 if road.is_visited else 0, # whether road has been previously visited
				) if euclidean_distance(road.start.pos,j.pos) < euclidean_distance(road.end.pos,j.pos) else (
					*shift_rotate_normalise_point(road.end.pos),
					*shift_rotate_normalise_point(road.start.pos),
					*road.binary_features(as_tuple=True), # in [0,1]
					1 if road.is_visited else 0, # whether road has been previously visited
				)
				for road in j.roads_connected
			], key=lambda x:(x[0:4])) + [ # placeholders for unavailable roads
				(
					-1,-1,-1,-1,
					*[-1]*self.obs_road_features,
					-1,
				)
			]*(self.max_roads_per_junction-len(j.roads_connected))
			for j,_ in sorted(zip((j1,j2),relative_road_points),key=lambda x:x[1])
		], dtype=np.float32)
		# print(junction_view.shape)
		return road_view, junction_view
	
	def reset(self):
		self.is_over = False
		self.episode_statistics = {}
		self._step = 0
		###########################
		self.seconds_per_step = self.get_step_seconds()
		# car position
		self.car_point = self.road_network.set(self.junction_number)
		self.car_orientation = (2*np.random.random()-1)*np.pi # in [-pi,pi]
		self.distance_to_closest_road, self.closest_road, self.closest_junctions = self.road_network.get_closest_road_and_junctions(self.car_point)
		self.last_closest_road = None
		# steering angle & speed
		self.speed = self.min_speed # self.min_speed + (self.max_speed-self.min_speed)*np.random.random() # in [min_speed,max_speed]
		# self.speed = self.min_speed+(self.max_speed-self.min_speed)*(70/120) # for testing
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
		return np.random.exponential(scale=self.mean_seconds_per_step) if self.random_seconds_per_step is True else self.mean_seconds_per_step

	def is_in_junction(self, car_point):
		distance_from_junction = min(euclidean_distance(self.closest_junctions[0].pos, car_point), euclidean_distance(self.closest_junctions[1].pos, car_point))
		return distance_from_junction <= self.junction_radius

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
		# if a new road is visited, add the old one to the set of visited ones	
		if self.last_closest_road != self.closest_road and not self.is_in_junction(self.car_point):
			visiting_new_road = True
			if self.last_closest_road is not None: # if closest_road is not the first visited road
				self.last_closest_road.is_visited = True
			self.last_closest_road = self.closest_road # keep track of the current road
		else:
			visiting_new_road = False
		# compute perceived reward
		reward, dead, reward_type = self.get_reward(
			car_speed=self.speed, 
			car_point=self.car_point, 
			old_car_point=old_car_point, 
			visiting_new_road=visiting_new_road,
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
		out_of_time = self._step >= self.max_step
		terminal = dead or out_of_time
		info_dict = {'explanation':reward_type}
		if terminal: # populate statistics
			self.is_over = True
			stats = {
				"avg_speed": self.avg_speed_per_steps/self._step,
				"out_of_time": 1 if out_of_time else 0,
			}
			info_dict.update(stats)
			self.episode_statistics = stats
		return [state, reward, terminal, info_dict]
			
	def get_info(self):
		return f"speed={self.speed}, steering_angle={self.steering_angle}, orientation={self.car_orientation}\n"
		
	def get_screen(self): # RGB array
		# First set up the figure and the axis
		# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
		figure = Figure(figsize=(5,5), tight_layout=True)
		canvas = FigureCanvas(figure)
		ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
		
		# [Junctions]
		if len(self.road_network.junctions) > 0:
			junctions = [Circle(junction.pos, self.junction_radius, color='y', alpha=0.25) for junction in self.road_network.junctions]
			patch_collection = PatchCollection(junctions, match_original=True)
			ax.add_collection(patch_collection)

		# [Car]
		car_x, car_y = self.car_point
		car_handle = ax.scatter(car_x, car_y, marker='o', color='g', label='Car')
		# [Heading Vector]
		dir_x, dir_y = get_heading_vector(angle=self.car_orientation, space=self.max_dimension/16)
		heading_vector_handle, = ax.plot([car_x, car_x+dir_x],[car_y, car_y+dir_y], color='g', alpha=0.5, label='Heading Vector')

		# [Roads]
		for road in self.road_network.roads:
			road_pos = list(zip(*(road.start.pos, road.end.pos)))
			# print("Drawing road {} {}".format(road[0], road[1]))
			if road.colour is None:
				min_speed = self.road_network.road_culture.get_minimum_speed(road, self.road_network.agent) # None if road is unfeasible
				can_move = min_speed is not None
				road.colour = "Green" if can_move else "Red"
			road_colour = road.colour
			if road_colour == "Green":
				self.road_network.agent.assign_property_value("Speed", self.road_network.normalise_speed(self.min_speed, self.max_speed, self.speed))
				correct_properties, _ = self.road_network.run_dialogue(road, self.road_network.agent, explanation_type="compact")
				if not correct_properties:
					road_colour = "Gold"
			path_handle, = ax.plot(road_pos[0], road_pos[1], color=colour_to_hex(road_colour), ls='--' if road==self.closest_road else '-', lw=2, alpha=0.5, label="Road")
			# ax.fill_between(road_pos[0], np.array(road_pos[1])+self.max_distance_to_path, np.array(road_pos[1])-self.max_distance_to_path, alpha=0.1, color=colour_to_hex(road_colour))
			# ax.fill_between(road_pos[1], np.array(road_pos[0])+self.max_distance_to_path, np.array(road_pos[0])-self.max_distance_to_path, alpha=0.1, color=colour_to_hex(road_colour))

		path1_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Green"), lw=2, alpha=0.5, label="OK")
		path2_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Red"), lw=2, alpha=0.5, label="Unfeasible")
		path3_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Gold"), lw=2, alpha=0.5, label="Wrong Speed")
		path4_handle, = ax.plot((0,0), (0,0), color=colour_to_hex("Black"), ls='--', lw=2, alpha=0.5, label="Current Road")
		# junction_handle = ax.scatter(0, 0, marker='o', color='y', label='Junction')

		# Adjust ax limits in order to get the same scale factor on both x and y
		a,b = ax.get_xlim()
		c,d = ax.get_ylim()
		max_length = max(d-c, b-a)
		ax.set_xlim([a,a+max_length])
		ax.set_ylim([c,c+max_length])
		# Build legend
		handles = [car_handle, path1_handle, path2_handle, path3_handle, path4_handle]
		ax.legend(handles=handles)
		# Draw plot
		figure.suptitle(' '.join([
			f'[Angle]{convert_radiant_to_degree(self.steering_angle):.2f}°', 
			f'[Orient.]{convert_radiant_to_degree(self.car_orientation):.2f}°', 
			f'[Speed]{self.speed:.2f} m/s', 
			'\n',
			f'[Step]{self._step}', 
			f'[Old]{self.closest_road.is_visited}', 
			f'[Car]{self.road_network.agent.binary_features()}', 
			f'[Reward]{self.last_reward:.2f}',
		]))
		# figure.tight_layout()
		canvas.draw()
		# Save plot into RGB array
		data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
		figure.clear()
		return data # RGB array

	def render(self, mode='human'):
		img = self.get_screen()
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen
				
	def get_statistics(self):
		return self.episode_statistics
