import numpy as np

from experience_buffers.replay_buffer import LocalReplayBuffer
from experience_buffers.clustering_scheme import *

def get_clustered_replay_buffer(config):
	assert config["batch_mode"] == "complete_episodes", "This algorithm requires 'complete_episodes' as batch_mode"
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		priority_weights_key=config["priority_weight"],
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
		replay_sequence_length=config["replay_sequence_length"], 
		priority_weights_aggregator=config["priority_weights_aggregator"], 
	)
	clustering_scheme = eval(config["clustering_scheme"])()
	return local_replay_buffer, clustering_scheme

def assign_types_from_episode(episode, clustering_scheme):
	episode_type = clustering_scheme.get_episode_type(episode)
	for batch in episode:
		batch_type = clustering_scheme.get_batch_type(batch, episode_type)
		batch["batch_types"] = np.array([batch_type]*batch.count)
	return episode

