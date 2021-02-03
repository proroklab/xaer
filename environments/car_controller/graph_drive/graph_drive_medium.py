
from environments.car_controller.graph_drive.graph_drive_easy import GraphDriveEasy
from environments.car_controller.grid_drive.lib.road_cultures import MediumRoadCulture

class GraphDriveMedium(GraphDriveEasy):
	CULTURE = MediumRoadCulture
