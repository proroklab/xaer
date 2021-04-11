
from environments.car_controller.graph_drive.sparse_graph_drive_easy import SparseGraphDriveEasy
from environments.car_controller.grid_drive.lib.road_cultures import HardRoadCulture

class SparseGraphDriveHard(SparseGraphDriveEasy):
	CULTURE = HardRoadCulture
