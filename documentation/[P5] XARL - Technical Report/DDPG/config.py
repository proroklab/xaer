experiment_list = [
# P1 - default: XADDPG, batch_mode = 'complete_episodes', prioritized_drop_probability = 0.5, clustering_scheme = 'moving_best_extrinsic_reward_with_multiple_types'
	"gualtiero", # DDPG
	+"dalibor", # <default>
	-"mingo", # prioritised_cluster_sampling = False
	"gretel", # clustering_scheme = 'none'
	+"moschina", # clustering_scheme = 'moving_best_extrinsic_reward'
	"benes", # clustering_scheme = 'extrinsic_reward'
	"donprocopio", # clustering_scheme = 'reward_with_multiple_types'
	-"ines", # clustering_scheme = 'reward_with_type'
	-"dorina", # clustering_scheme = 'moving_best_extrinsic_reward_with_type'
	"remendado", # global_distribution_matching = True
	++"zuniga", # beta = None
	"donandronico", # prioritized_drop_probability = 0
	--"grenvil", # prioritized_drop_probability = 1
# P2 - default: XAPPO, batch_mode = 'complete_episodes', prioritized_drop_probability = 0, clustering_scheme = 'moving_best_extrinsic_reward_with_multiple_types', replay_proportion = 1, clip_param = 0.2, gae_with_vtrace = False
	"filindo", # APPO
	-"malatesta", # APPO, replay_proportion = 0
	++"donpasquale", # <default>
	+"doncurzio", # prioritised_cluster_sampling = False
	"bettina", # clustering_scheme = 'none'
	"brander", # clustering_scheme = 'moving_best_extrinsic_reward'
	"fiorello", # clustering_scheme = 'extrinsic_reward'
	"lucia", # clustering_scheme = 'reward_with_multiple_types'
	"morales", # clustering_scheme = 'reward_with_type'
	"edmondo", # clustering_scheme = 'moving_best_extrinsic_reward_with_type'
	"berta", # global_distribution_matching = True
	"hansel", # # prioritized_replay = False
	+"marullo", # replay_proportion = 2
	"leonora", # replay_proportion = 0.5
	"lily", # replay_proportion = 0
	-"eufemia", # gae_with_vtrace = True
	+"eboli", # priority_id = advantages
	+"pancrazio", # priority_id = rewards
	"giovanna", # prioritized_drop_probability = 0.5
	--"donbasilio", # prioritized_drop_probability = 1
]
