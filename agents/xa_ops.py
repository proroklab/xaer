import numpy as np

from experience_buffers.replay_buffer import LocalReplayBuffer
from experience_buffers.clustering_scheme import *
from ray.rllib.execution.learner_thread import LearnerThread, get_learner_stats
from ray.rllib.execution.multi_gpu_learner import TFMultiGPULearner, get_learner_stats as get_gpu_learner_stats

def get_clustered_replay_buffer(config):
	assert config["batch_mode"] == "complete_episodes" or config["clustering_scheme"] not in ["moving_best_extrinsic_reward_with_type","extrinsic_reward"], f"This algorithm requires 'complete_episodes' as batch_mode when 'clustering_scheme' is {config['clustering_scheme']}"
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
		replay_sequence_length=config["replay_sequence_length"], 
	)
	clustering_scheme = eval(config["clustering_scheme"])()
	return local_replay_buffer, clustering_scheme

def assign_types_from_episode(episode, clustering_scheme):
	episode_type = clustering_scheme.get_episode_type(episode)
	for batch in episode:
		batch_type = clustering_scheme.get_batch_type(batch, episode_type)
		batch["batch_types"] = np.array([batch_type]*batch.count)
	return episode

class BatchLearnerThread(LearnerThread):
    def step(self):
        with self.queue_timer:
            batch, _ = self.minibatch_buffer.get()

        with self.grad_timer:
            fetches = self.local_worker.learn_on_batch(batch)
            self.weights_updated = True
            self.stats = get_learner_stats(fetches)

        self.num_steps += 1
        self.outqueue.put((batch, self.stats))
        self.learner_queue_size.push(self.inqueue.qsize())

class BatchTFMultiGPULearner(TFMultiGPULearner):
    def step(self):
        assert self.loader_thread.is_alive()
        with self.load_wait_timer:
            opt, released = self.minibatch_buffer.get()

        with self.grad_timer:
            fetches = opt.optimize(self.sess, 0)
            self.weights_updated = True
            self.stats = get_gpu_learner_stats(fetches)

        if released:
            self.idle_optimizers.put(opt)

        self.outqueue.put((opt, self.stats))
        self.learner_queue_size.push(self.inqueue.qsize())

def xa_make_learner_thread(local_worker, config):
    if config["num_gpus"] > 1 or config["num_data_loader_buffers"] > 1:
        logger.info(
            "Enabling multi-GPU mode, {} GPUs, {} parallel loaders".format(
                config["num_gpus"], config["num_data_loader_buffers"]))
        if config["num_data_loader_buffers"] < config["minibatch_buffer_size"]:
            raise ValueError(
                "In multi-gpu mode you must have at least as many "
                "parallel data loader buffers as minibatch buffers: "
                "{} vs {}".format(config["num_data_loader_buffers"],
                                  config["minibatch_buffer_size"]))
        learner_thread = TFMultiGPULearner(
            local_worker,
            num_gpus=config["num_gpus"],
            lr=config["lr"],
            train_batch_size=config["train_batch_size"],
            num_data_loader_buffers=config["num_data_loader_buffers"],
            minibatch_buffer_size=config["minibatch_buffer_size"],
            num_sgd_iter=config["num_sgd_iter"],
            learner_queue_size=config["learner_queue_size"],
            learner_queue_timeout=config["learner_queue_timeout"])
    else:
        learner_thread = BatchLearnerThread(
            local_worker,
            minibatch_buffer_size=config["minibatch_buffer_size"],
            num_sgd_iter=config["num_sgd_iter"],
            learner_queue_size=config["learner_queue_size"],
            learner_queue_timeout=config["learner_queue_timeout"])
    return learner_thread

