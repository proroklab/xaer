import numpy as np
from environment.discrete_environment.road_cell import RoadCell
from environment.discrete_environment.road_agent import RoadAgent
from environment.discrete_environment.road_cultures import *
import copy

class RoadGrid:
    def __init__(self, x_dim, y_dim):
        self.agent = RoadAgent()
        self.agent_position = (0, 0)
        self.cells = []
        self.visited_positions = set()
        self.width = x_dim
        self.height = y_dim

        self.road_culture = MediumRoadCulture()
        self.agent.set_culture(self.road_culture)
        self.road_culture.initialise_random_agent(self.agent)

        self.initialise_random_grid()

    def within_bounds(self, coord):
        """
        Checks if a given coordinate exists in the 2D grid.
        :param coord: Position to check.
        :returns True if within bounds. False otherwise.
        """
        wr = range(0, self.width)
        hr = range(0, self.height)
        x, y = coord
        return x in wr and y in hr

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
            if self.within_bounds((x - 1, y)): neighbours.append((x - 1, y))  # Left
            if self.within_bounds((x + 1, y)): neighbours.append((x + 1, y))  # Right
            if self.within_bounds((x, y - 1)): neighbours.append((x, y - 1))  # Up
            if self.within_bounds((x, y + 1)): neighbours.append((x, y + 1))  # Down
        elif neighbourhood_type == 'moore':
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == j == 0: continue
                    if self.within_bounds((x + i, y + j)): neighbours.append((x + i, y + j))
        else:
            print("RoadGrid::neighbours_of: This neighbourhood type is not supported.")
            return None
        return neighbours

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

    def move_agent(self, dest_coord, speed):
        """
        Attempts to move an agent to a neighbouring cell.
        :param speed: commanded speed to traverse next cell
        :param dest_coord: destination RoadCell
        :return: False if move is illegal. Integer-valued reward if move is valid.
        """
        current_coord = self.agent_position
        if current_coord is False:
            print("RoadGrid::move_agent: Agent not found!")
            return False

        reward = 0

        self.visited_positions.add(current_coord)

        x, y = dest_coord
        temp_AF = copy.deepcopy(self.road_culture.AF)
        self.agent.assign_property_value("Speed", speed)

        # Prune temporary AF out of unverified arguments
        to_remove = []
        for argument_id in temp_AF.all_arguments:
            argument_obj = temp_AF.argument(argument_id)
            if not argument_obj.verify(self.cells[x][y], self.agent):
                to_remove.append(argument_id)
        for argument_id in to_remove:
            temp_AF.remove_argument(argument_id)

        # Check if motion belongs to grounded extension.
        solver_result = temp_AF.run_solver(semantics="DS-PR", arg_str="0")
        if "YES" in solver_result:  # If the argument "I will not get a ticket" is valid
            reward += 5

        # Check if not repeating previously-visited cells.
        if (x, y) not in self.visited_positions:
            reward += 2

        self.agent_position = (x, y)

        return reward



