from ray.tune.registry import register_env
######### Add new environment below #########

from environments.gym_env_example import Example_v0
register_env("ToyExample-v0", lambda config: Example_v0())

from environments.car_controller.car_controller_v1 import CarControllerV1
register_env("CescoDrive-v0", lambda config: CarControllerV1())

from environments.car_controller.car_controller_v2 import CarControllerV2
register_env("CescoDrive-v1", lambda config: CarControllerV2())

from environments.car_controller.car_controller_v3 import CarControllerV3
register_env("CescoDrive-v2", lambda config: CarControllerV3())

from environments.car_controller.car_controller_v4 import CarControllerV4
register_env("AlexDrive-v0", lambda config: CarControllerV4())