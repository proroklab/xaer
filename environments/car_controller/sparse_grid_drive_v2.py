# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v1 import GridDriveV1

class SparseGridDriveV2(GridDriveV1):

	def reset(self):
		self.cumulated_return = 0
		return super().reset()

	def step(self, action_vector):
		direction = action_vector//self.MAX_GAPPED_SPEED
		gapped_speed = action_vector%self.MAX_GAPPED_SPEED
		self.step_counter += 1
		x, y = self.grid.agent_position
		self.grid_view[x][y][-1] = 0 # remove old position
		speed = gapped_speed*self.SPEED_GAP
		can_move, explanation = self.grid.move_agent(direction, speed)
		x, y = self.grid.agent_position
		if can_move:
			if self.grid_view[x][y][-2] > 0: # already visited cell
				explanation = 'Old cell'
			else:
				self.cumulated_return += (speed+1)/self.MAX_SPEED # in (0,1]
				explanation = 'OK'
		# do it aftwer checking positions
		self.grid_view[x][y][-2] = 1 # set current cell as visited
		self.grid_view[x][y][-1] = 1 # set new position
		is_terminal_step = self.step_counter >= self.MAX_STEP
		return [
			self.get_state(), # observation
			self.cumulated_return if is_terminal_step and can_move else 0, # reward
			is_terminal_step or not can_move, # terminal
			{'explanation': explanation} # info_dict
		]
