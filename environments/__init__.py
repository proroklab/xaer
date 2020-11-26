from environments.gym_env_example import *
from ray.tune.registry import register_env

register_env("example-v0", lambda config: Example_v0())
