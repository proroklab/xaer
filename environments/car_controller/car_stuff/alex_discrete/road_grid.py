# import numpy as np
from environments.car_controller.car_stuff.alex_discrete.road_cell import RoadCell
from environments.car_controller.car_stuff.alex_discrete.road_agent import RoadAgent
from random import randrange

NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3

class RoadGrid:
	def __init__(self, x_dim, y_dim, culture):
		self.agent = RoadAgent()
		self.agent_position = (0, 0)
		self.cells = []
		self.width = x_dim
		self.height = y_dim

		self.road_culture = culture
		self.agent.set_culture(self.road_culture)
		self.road_culture.initialise_random_agent(self.agent)
		# self.inaccessible = tuple([0] * (len(self.road_culture.properties) + 1))

		self.initialise_random_grid()

	def set_random_position(self):
		self.agent_position = (randrange(0,self.width), randrange(0,self.height))

	def within_bounds(self, coord):
		"""
		Checks if a given coordinate exists in the 2D grid.
		:param coord: Position to check.
		:returns True if within bounds. False otherwise.
		"""
		return 0 <= coord[0] < self.width and 0 <= coord[1] < self.height

	def neighbours_of(self, coord, neighbourhood_type='von_neumann'):
		"""
		Returns immediate neighbourhood of given coordinate.
		:param coord: Position to return neighbourhood.
		:param neighbourhood_type: Currently supports 'von_neumann' or 'moore' neighbourhoods.
		:return: List of neighbours. False if arguments are invalid..
		"""
		if not self.within_bounds(coord):
			print("RoadGrid::neighbours_of: Position out of bounds.")
			return None
		x, y = coord
		neighbours = []
		if neighbourhood_type == 'von_neumann':
			# if self.within_bounds((x - 1, y)): neighbours.append((x - 1, y))  # Left
			# if self.within_bounds((x + 1, y)): neighbours.append((x + 1, y))  # Right
			# if self.within_bounds((x, y - 1)): neighbours.append((x, y - 1))  # Up
			# if self.within_bounds((x, y + 1)): neighbours.append((x, y + 1))  # Down
			neighbours += [
				((x - 1)%self.height, y), # Left
				((x + 1)%self.height, y), # Right
				(x, (y - 1)%self.width), # Up
				(x, (y + 1)%self.width), # Down
			]
		elif neighbourhood_type == 'moore':
			for i in range(-1, 2):
				for j in range(-1, 2):
					if i == j == 0: continue
					# if self.within_bounds((x + i, y + j)): neighbours.append((x + i, y + j))
					neighbours.append(((x + i)%self.height, (y + j)%self.width))
		else:
			print("RoadGrid::neighbours_of: This neighbourhood type is not supported.")
			return None
		return neighbours

	def neighbour_features(self):
		# Start with order NORTH, SOUTH, EAST, WEST.
		x, y = self.agent_position
		north_features = self.cells[x][(y + 1)%self.width].binary_features() #if self.within_bounds((x, y + 1)) else self.inaccessible
		south_features = self.cells[x][(y - 1)%self.width].binary_features() #if self.within_bounds((x, y - 1)) else self.inaccessible
		east_features  = self.cells[(x + 1)%self.height][y].binary_features() #if self.within_bounds((x + 1, y)) else self.inaccessible
		west_features  = self.cells[(x - 1)%self.height][y].binary_features() #if self.within_bounds((x - 1, y)) else self.inaccessible

		total_features = north_features + south_features + east_features + west_features
		return total_features

	def get_features(self):
		return [
			[
				e.binary_features()
				for e in row
			]
			for row in self.cells
		]

	def initialise_random_grid(self):
		"""
		Fills a grid with random RoadCells, each initialised by the current culture.
		:return:
		"""
		for i in range(self.width):
			self.cells.append([])
			for j in range(self.height):
				road = RoadCell(i, j)
				road.set_culture(self.road_culture)
				self.road_culture.initialise_random_road(road)
				self.cells[i].append(road)

	def run_dialogue(self, road, agent, explanation_type="verbose"):
		"""
		Runs dialogue to find out decision regarding penalty in argumentation framework.
		Args:
			road: RoadCell corresponding to destination cell.
			agent: RoadAgent corresponding to agent.
			explanation_type: 'verbose' for all arguments used in exchange; 'compact' for only winning ones.

		Returns: Decision on penalty + explanation.
		"""
		# print("@@@@@@@@@@@@@ NEW DIALOGUE @@@@@@@@@@@@@")
		AF = self.road_culture.AF
		verified = set()

		# Prune temporary AF out of unverified arguments
		to_remove = []
		for argument_id in AF.all_arguments:
			argument_obj = AF.argument(argument_id)
			if argument_obj.verify(road, agent):
				verified.add(argument_id)

		# Game starts with proponent using argument 0 ("I will not get a ticket").
		used_arguments = {"opponent": set(), "proponent": {0}}
		last_argument = {"opponent": set(), "proponent": {0}}

		dialogue_history = []
		dialogue_history.append(last_argument["proponent"])

		# Odd turns: opponent. Even turns: proponent.
		turn = 1
		game_over = False
		winner = None
		while not game_over:
			# print("##### TURN {} #####".format(turn))
			if turn % 2:
				# Opponent's turn.
				current_player = "opponent"
				next_player = "proponent"
			else:
				# Proponent's turn.
				current_player = "proponent"
				next_player = "opponent"
			turn += 1
			# Remove previously used arguments.
			all_used_arguments = used_arguments["proponent"] | used_arguments["opponent"]
			forbidden_arguments = set(all_used_arguments)
			# Cannot pick argument that is attacked by previously used argument.
			forbidden_arguments.update(AF.arguments_attacked_by_list(list(all_used_arguments)))
			# print("All used arguments: {}".format(all_used_arguments))
			# print("Forbidden arguments: {}".format(forbidden_arguments))
			# Use all arguments as possible.
			all_viable_arguments = set(AF.arguments_that_attack(list(last_argument[next_player])))
			# print("Viable arguments: {}".format(all_viable_arguments))
			verified_attacks = verified.intersection(all_viable_arguments)
			# print("Verified attacks: {}".format(verified_attacks))
			targets = set(AF.arguments_attacked_by_list(list(verified_attacks)))
			if last_argument[next_player].issubset(targets):
				used_arguments[current_player].update(verified_attacks)
				last_argument[current_player] = verified_attacks
				# print("{} used arguments {}".format(current_player, verified_attacks))
				dialogue_history.append(verified_attacks)
			else:
				game_over = True
				winner = next_player
				# print("GAME OVER! {} wins".format(winner))

		motion_validated = True if winner == "proponent" else False

		# Building the explanation.

		if explanation_type == "verbose":
			turn = 0
			explanation_list = []
			for argument_list in dialogue_history:
				argument_explanation = "CON: " if turn % 2 else "PRO: "
				argument_explanation += ' / '.join(sorted((
					AF.argument(argument_id).descriptive_text
					for argument_id in argument_list
				)))
				turn += 1
				explanation_list.append(argument_explanation)
		else:
			explanation_list = [
				AF.argument(argument_id).descriptive_text
				for argument_id in last_argument[winner]
			]

		return motion_validated, explanation_list

	def move_agent(self, direction, speed):
		"""
		Attempts to move an agent to a neighbouring cell.
		:param speed: commanded speed to traverse next cell
		:param direction: 0 == NORTH, 1 == SOUTH, 2 == EAST, 3 == WEST
		:return: False if move is illegal. Integer-valued reward if move is valid.
		"""
		# if self.agent_position is False:
		# 	print("RoadGrid::move_agent: Agent not found!")
		# 	return False

		dest_x, dest_y = self.agent_position
		if   direction == NORTH:
			dest_y += 1
		elif direction == SOUTH:
			dest_y -= 1
		elif direction == EAST:
			dest_x += 1
		elif direction == WEST:
			dest_x -= 1
		dest_x %= self.width # infinite grid
		dest_y %= self.height # infinite grid

		# if not self.within_bounds((dest_x, dest_y)):
		# 	return "FAIL: Out of bounds!"

		self.agent_position = (dest_x, dest_y)
		self.agent.assign_property_value("Speed", speed)

		can_move, explanation_list = self.run_dialogue(self.cells[dest_x][dest_y], self.agent, explanation_type="compact")
		return can_move, explanation_list

