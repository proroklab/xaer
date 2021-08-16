from ray.tune.registry import register_env
######### Add new environment below #########

from environments.custom_metrics import CustomEnvironmentCallbacks

from environments.gym_env_example import Example_v0
register_env("ToyExample-V0", lambda config: Example_v0(config))

### CescoDrive
from environments.car_controller.cesco_drive.cesco_drive_v0 import CescoDriveV0
register_env("CescoDrive-V0", lambda config: CescoDriveV0(config))

from environments.car_controller.cesco_drive.cesco_drive_v1 import CescoDriveV1
register_env("CescoDrive-V1", lambda config: CescoDriveV1(config))

### GraphDrive
from environments.car_controller.graph_drive.graph_drive import GraphDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"GraphDrive-{culture_level}", lambda config: GraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V1", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V2", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V3", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v3', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-S*J", lambda config: GraphDrive({"reward_fn": 'frequent_reward_step_multiplied_by_junctions', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-FullStep", lambda config: GraphDrive({"reward_fn": 'frequent_reward_full_step', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse", lambda config: GraphDrive({"reward_fn": 'sparse_reward_default', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V1", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V2", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V3", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v3', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-S*J", lambda config: GraphDrive({"reward_fn": 'sparse_reward_step_multiplied_by_junctions', "culture_level": culture_level}))

### GridDrive
from environments.car_controller.grid_drive.grid_drive import GridDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"GridDrive-{culture_level}", lambda config: GridDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V1", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V2", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-S*J", lambda config: GridDrive({"reward_fn": 'frequent_reward_step_multiplied_by_junctions', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-FullStep", lambda config: GridDrive({"reward_fn": 'frequent_reward_full_step', "culture_level": culture_level}))

### Special Atari
from environments.special_atari import SpecialAtariEnv
for game in [
	"adventure",
	"air_raid",
	"alien",
	"amidar",
	"assault",
	"asterix",
	"asteroids",
	"atlantis",
	"bank_heist",
	"battle_zone",
	"beam_rider",
	"berzerk",
	"bowling",
	"boxing",
	"breakout",
	"carnival",
	"centipede",
	"chopper_command",
	"crazy_climber",
	"defender",
	"demon_attack",
	"double_dunk",
	"elevator_action",
	"enduro",
	"fishing_derby",
	"freeway",
	"frostbite",
	"gopher",
	"gravitar",
	"hero",
	"ice_hockey",
	"jamesbond",
	"journey_escape",
	"kangaroo",
	"krull",
	"kung_fu_master",
	"montezuma_revenge",
	"ms_pacman",
	"name_this_game",
	"phoenix",
	"pitfall",
	"pong",
	"pooyan",
	"private_eye",
	"qbert",
	"riverraid",
	"road_runner",
	"robotank",
	"seaquest",
	"skiing",
	"solaris",
	"space_invaders",
	"star_gunner",
	"tennis",
	"time_pilot",
	"tutankham",
	"up_n_down",
	"venture",
	"video_pinball",
	"wizard_of_wor",
	"yars_revenge",
	"zaxxon",
]:
	for obs_type in ["image", "ram"]:
		# space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
		name = "".join([g.capitalize() for g in game.split("_")])
		if obs_type == "ram":
			name = "{}-ram".format(name)

		nondeterministic = False
		if game == "elevator_action" and obs_type == "ram":
			# ElevatorAction-ram-v0 seems to yield slightly
			# non-deterministic observations about 10% of the time. We
			# should track this down eventually, but for now we just
			# mark it as nondeterministic.
			nondeterministic = True

		register_env(
			"Special{}-v0".format(name),
			lambda config: SpecialAtariEnv(**{
				"game": game,
				"obs_type": obs_type,
				"repeat_action_probability": 0.25,
			})
		)

		register_env(
			"Special{}-v4".format(name),
			lambda config: SpecialAtariEnv(**{"game": game, "obs_type": obs_type})
		)

		# Standard Deterministic (as in the original DeepMind paper)
		if game == "space_invaders":
			frameskip = 3
		else:
			frameskip = 4

		# Use a deterministic frame skip.
		register_env(
			"Special{}Deterministic-v0".format(name),
			lambda config: SpecialAtariEnv(**{
				"game": game,
				"obs_type": obs_type,
				"frameskip": frameskip,
				"repeat_action_probability": 0.25,
			})
		)

		register_env(
			"Special{}Deterministic-v4".format(name),
			lambda config: SpecialAtariEnv(**{"game": game, "obs_type": obs_type, "frameskip": frameskip})
		)

		register_env(
			"Special{}NoFrameskip-v0".format(name),
			lambda config: SpecialAtariEnv(**{
				"game": game,
				"obs_type": obs_type,
				"frameskip": 1,
				"repeat_action_probability": 0.25,
			})
		)

		# No frameskip. (Atari has no entropy source, so these are
		# deterministic environments.)
		register_env(
			"Special{}NoFrameskip-v4".format(name),
			lambda config: SpecialAtariEnv(**{
				"game": game,
				"obs_type": obs_type,
				"frameskip": 1,
			})
		)