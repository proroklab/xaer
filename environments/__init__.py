from ray.tune.registry import register_env
######### Add new environment below #########

from environments.gym_env_example import Example_v0
register_env("ToyExample-V0", lambda config: Example_v0())

### CescoDrive
from environments.car_controller.cesco_drive.cesco_drive_v0 import CescoDriveV0
register_env("CescoDrive-V0", lambda config: CescoDriveV0())

from environments.car_controller.cesco_drive.cesco_drive_v1 import CescoDriveV1
register_env("CescoDrive-V1", lambda config: CescoDriveV1())

### GraphDrive
from environments.car_controller.graph_drive.graph_drive_easy import GraphDriveEasy
register_env("GraphDrive-Easy", lambda config: GraphDriveEasy())

from environments.car_controller.graph_drive.graph_drive_medium import GraphDriveMedium
register_env("GraphDrive-Medium", lambda config: GraphDriveMedium())

from environments.car_controller.graph_drive.graph_drive_hard import GraphDriveHard
register_env("GraphDrive-Hard", lambda config: GraphDriveHard())

### GridDrive
from environments.car_controller.grid_drive.grid_drive_easy import GridDriveEasy
register_env("GridDrive-Easy", lambda config: GridDriveEasy())

from environments.car_controller.grid_drive.grid_drive_medium import GridDriveMedium
register_env("GridDrive-Medium", lambda config: GridDriveMedium())

from environments.car_controller.grid_drive.grid_drive_hard import GridDriveHard
register_env("GridDrive-Hard", lambda config: GridDriveHard())

from environments.car_controller.grid_drive.sparse_grid_drive_hard_v1 import SparseGridDriveHardV1
register_env("Sparse-GridDrive-Hard-V1", lambda config: SparseGridDriveHardV1())

from environments.car_controller.grid_drive.sparse_grid_drive_hard_v2 import SparseGridDriveHardV2
register_env("Sparse-GridDrive-Hard-V2", lambda config: SparseGridDriveHardV2())
