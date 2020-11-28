"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501

from ray.rllib.agents.dqn.dqn import *
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from agents.xa_ops import *
from experience_buffers.replay_ops import StoreToReplayBuffer

XADQN_DEFAULT_CONFIG = DEFAULT_CONFIG
XADQN_DEFAULT_CONFIG["buffer_options"] = {
    'size': 2**9, 
    'alpha': 0.5, 
    'prioritized_drop_probability': 0.5, 
    'global_distribution_matching': False, 
    'prioritised_cluster_sampling': True,
}
XADQN_DEFAULT_CONFIG["worker_side_prioritization"] = True
XADQN_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADQN_DEFAULT_CONFIG["batch_mode"] = "complete_episodes"
XADQN_DEFAULT_CONFIG["priority_weight"] = "weights" # one of the following: weigths, rewards, prev_rewards, action_logp
XADQN_DEFAULT_CONFIG["priority_weights_aggregator"] = 'np.mean' # a reduce function (from a list of numbers to a number)

def xadqn_execution_plan(workers, config):
    assert config["worker_side_prioritization"], "Worker side prioritization must be enabled"
    local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
    def update_priorities(item):
        batch, info_dict = item
        # print(info_dict)
        if config.get("prioritized_replay"):
            if config["priority_weight"] == "weights":
                weights = info_dict[DEFAULT_POLICY_ID].get("td_error", info_dict[DEFAULT_POLICY_ID][LEARNER_STATS_KEY].get("td_error"))
            else:
                weights = batch[config["priority_weight"]]
            local_replay_buffer.update_priority(
                batch_index=batch["batch_indexes"][0], 
                weights=weights, 
                type_id=batch["batch_types"][0],
            )
        return info_dict

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our local replay buffer. Calling
    # next() on store_op drives this.
    store_op = rollouts \
        .for_each(lambda batch: batch.split_by_episode()) \
        .flatten() \
        .for_each(lambda episode: episode.timeslices(config["rollout_fragment_length"])) \
        .for_each(lambda episode: assign_types_from_episode(episode, clustering_scheme)) \
        .flatten() \
        .for_each(StoreToReplayBuffer(local_buffer=local_replay_buffer))

    # (2) Read and train on experiences from the replay buffer. Every batch
    # returned from the LocalReplay() iterator is passed to TrainOneStep to
    # take a SGD step, and then we decide whether to update the target network.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(lambda x: post_fn(x, workers, config)) \
        .for_each(TrainOneStep(workers)) \
        .for_each(update_priorities) \
        .for_each(UpdateTargetNetwork(workers, config["target_network_update_freq"]))

    # Alternate deterministically between (1) and (2). Only return the output
    # of (2) since training metrics are not available until (2) runs.
    train_op = Concurrently([store_op, replay_op], mode="round_robin", output_indexes=[1], round_robin_weights=calculate_rr_weights(config))

    return StandardMetricsReporting(train_op, workers, config)

XADQNTrainer = DQNTrainer.with_updates(
    name="XADQN", 
    execution_plan=xadqn_execution_plan,
)
