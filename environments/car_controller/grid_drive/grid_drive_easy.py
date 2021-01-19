# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive.grid_drive_hard import GridDriveHard
from environments.car_controller.grid_drive.lib.road_cultures import EasyRoadCulture


class GridDriveEasy(GridDriveHard):
	CULTURE = EasyRoadCulture