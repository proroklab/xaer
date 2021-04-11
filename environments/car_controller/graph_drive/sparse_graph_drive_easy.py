# -*- coding: utf-8 -*-
from environments.car_controller.graph_drive.graph_drive_easy import GraphDriveEasy

class SparseGraphDriveEasy(GraphDriveEasy):
	
	def get_reward(self, visiting_new_road, old_goal_junction, old_car_point): # to finish
		def null_reward(is_terminal, label):
			return (0, is_terminal, label)
		def unitary_reward(is_positive, is_terminal, label):
			return (1 if is_positive else -1, is_terminal, label)
		def step_reward(is_positive, is_terminal, label):
			# reward = (np.mean(self.current_road_speed_list) - self.min_speed*0.9)/(self.max_speed-self.min_speed*0.9) # in (0,1]
			reward = len(self.visited_junctions)
			return (reward if is_positive else -reward, is_terminal, label)
		explanation_list_with_label = lambda _label,_explanation_list: list(map(lambda x:(_label,x), _explanation_list)) if _explanation_list else _label

		is_in_junction = self.is_in_junction(self.car_point)
		if is_in_junction:
			#######################################
			# "Is in new junction" rule
			if self.acquired_junction:  # If agent acquired a brand new junction.
				# return step_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
				return unitary_reward(is_positive=True, is_terminal=False, label=explanation_list_with_label('is_in_new_junction', self.last_explanation_list))
			#######################################
			# "Is in old junction" rule
			return null_reward(is_terminal=False, label='is_in_old_junction')
		#######################################
		# "Stay on the road" rule
		if self.distance_to_closest_road >= self.max_distance_to_path:
			return unitary_reward(is_positive=False, is_terminal=True, label='not_staying_on_the_road')
		#######################################
		# "No u-turning outside junctions" rule
		space_traveled_towards_goal = euclidean_distance(self.goal_junction.pos, old_car_point) - euclidean_distance(self.goal_junction.pos, self.car_point) if self.goal_junction is not None else 0
		if space_traveled_towards_goal <= 0:
			return unitary_reward(is_positive=False, is_terminal=True, label='u_turning_outside_junction')
		#######################################
		# "Follow regulation" rule. # Run dialogue against culture.
		# Assign normalised speed to agent properties before running dialogues.
		following_regulation, explanation_list = self.road_network.run_dialogue(self.closest_road, self.road_network.agent, explanation_type="compact")
		if not following_regulation:
			return unitary_reward(is_positive=False, is_terminal=True, label=explanation_list_with_label('not_following_regulation',explanation_list))
		#######################################
		# "Move forward" rule
		self.last_explanation_list = explanation_list
		return null_reward(is_terminal=False, label=explanation_list_with_label('moving_forward',explanation_list))
