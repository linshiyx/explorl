from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import ray
import tensorflow as tf
from ray.rllib.agent import Agent
# from ray.rllib.ppo.rollout import collect_samples
from explorl.ppo.rollout import collect_samples
from ray.rllib.utils import FilterManager
from ray.tune.result import TrainingResult
from tensorflow.python import debug as tf_debug

from explorl.ppo.ppo_evaluator import PPOEvaluator
import datetime

DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Number of steps after which the rollout gets cut
    "horizon": 2000,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 1,
    # Stepsize of SGD
    "sgd_stepsize": 5e-5,
    # TODO(pcm): Expose the choice between gpus and cpus
    # as a command line argument.
    "devices": ["/cpu:%d" % i for i in range(4)],
    "tf_session_args": {
        "device_count": {"CPU": 4},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "intra_op_parallelism_threads": 1,
        "inter_op_parallelism_threads": 2,
    },
    # Batch size for policy evaluations for rollouts
    "rollout_batchsize": 1,
    # Total SGD batch size across all devices for SGD
    "sgd_batchsize": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Config params to pass to the model
    "model": {"free_log_std": False},
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch": 1000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "min_steps_per_task": 500,
    # Number of actors used to collect the rollouts
    "num_workers": 1,
    # Resource requirements for remote actors
    "worker_resources": {"num_cpus": 1},
    # Dump TensorFlow timeline after this many SGD minibatches
    "full_trace_nth_sgd_batch": -1,
    # Whether to profile data loading
    "full_trace_data_load": False,
    # Outer loop iteration index when we drop into the TensorFlow debugger
    "tf_debug_iteration": -1,
    # If this is True, the TensorFlow debugger is invoked if an Inf or NaN
    # is detected
    "tf_debug_inf_or_nan": False,
    # If True, we write tensorflow logs and checkpoints
    "write_logs": True,
    # Arguments to pass to the env creator
    "env_config": {},

    'num_batches': 1e6,
    'batches_per_save': 20,
    'batches_per_evaluate': 20,
    'num_evaluation': 1,
    'summary_id': datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S'),
}


class PPOAgent():
    _agent_name = "PPO"
    # _default_config = DEFAULT_CONFIG

    def __init__(self, env_id):
        self.config = DEFAULT_CONFIG
        self.logdir = "logs/{}/".format(self.config['summary_id'])

        self.global_step = 0
        self.kl_coeff = self.config["kl_coeff"]
        self.local_evaluator = PPOEvaluator(
            env_id, self.config, self.logdir, False)
        RemotePPOEvaluator = ray.remote(
            **self.config["worker_resources"])(PPOEvaluator)
        self.remote_evaluators = [
            RemotePPOEvaluator.remote(
                env_id, self.config, self.logdir, True)
            for _ in range(self.config["num_workers"])]
        self.start_time = time.time()
        if self.config["write_logs"]:
            self.file_writer = tf.summary.FileWriter(
                self.logdir, self.local_evaluator.sess.graph)
        else:
            self.file_writer = None
        self.saver = tf.train.Saver(max_to_keep=None)

    def run_init_op(self):
        self.local_evaluator.run_init_op()

    def train(self):
        agents = self.remote_evaluators
        config = self.config
        model = self.local_evaluator

        if (config["num_workers"] * config["min_steps_per_task"] >
                config["timesteps_per_batch"]):
            print(
                "WARNING: num_workers * min_steps_per_task > "
                "timesteps_per_batch. This means that the output of some "
                "tasks will be wasted. Consider decreasing "
                "min_steps_per_task or increasing timesteps_per_batch.")

        while self.global_step < self.config['num_batches']:

            iter_start = time.time()
            weights = ray.put(model.get_weights())
            [a.set_weights.remote(weights) for a in agents]

            samples = collect_samples(agents, config, self.local_evaluator)


            def standardized(value):
                # Divide by the maximum of value.std() and 1e-4
                # to guard against the case where all values are equal
                return (value - value.mean()) / max(1e-4, value.std())

            samples.data["advantages"] = standardized(samples["advantages"])

            rollouts_end = time.time()
            print("Computing policy (iterations=" + str(config["num_sgd_iter"]) +
                  ", stepsize=" + str(config["sgd_stepsize"]) + "):")
            samples.shuffle()
            shuffle_end = time.time()
            # tuples_per_device = model.load_data(
            #     samples, self.iteration == 0 and config["full_trace_data_load"])
            tuples_per_device = model.load_data(
                samples, config["full_trace_data_load"])
            load_end = time.time()
            rollouts_time = rollouts_end - iter_start
            shuffle_time = shuffle_end - rollouts_end
            load_time = load_end - shuffle_end
            sgd_time = 0
            for i in range(config["num_sgd_iter"]):
                sgd_start = time.time()
                batch_index = 0
                num_batches = (
                    int(tuples_per_device) // int(model.per_device_batch_size))
                permutation = np.random.permutation(num_batches)
                while batch_index < num_batches:
                    model.run_sgd_minibatch(permutation[batch_index] * model.per_device_batch_size,
                                            self.kl_coeff, False,
                                            self.file_writer)
                    batch_index += 1
                sgd_end = time.time()
                sgd_time += sgd_end - sgd_start

            self.global_step += 1
            # if kl > 2.0 * config["kl_target"]:
            #     self.kl_coeff *= 1.5
            # elif kl < 0.5 * config["kl_target"]:
            #     self.kl_coeff *= 0.5

            FilterManager.synchronize(
                self.local_evaluator.filters, self.remote_evaluators)

            info = {
                # "kl_divergence": kl,
                # "kl_coefficient": self.kl_coeff,
                "rollouts_time": rollouts_time,
                "shuffle_time": shuffle_time,
                "load_time": load_time,
                "sgd_time": sgd_time,
                "sample_throughput": len(samples["observations"]) / sgd_time
            }
            print(info)


            if self.global_step // self.config['batches_per_save'] == 0:
                self._save()

            if self.global_step // self.config['batches_per_evaluate'] == 0:
                pl, vl, ent, kl, e_pl, e_vl, e_ent, e_kl, rew, leng = \
                    self.local_evaluator.get_evaluate_metrics()

                stats = tf.Summary(value=[
                    tf.Summary.Value(tag="reward", simple_value=rew),
                    tf.Summary.Value(tag="episode_length", simple_value=leng),
                    tf.Summary.Value(tag="policy_loss", simple_value=pl),
                    tf.Summary.Value(tag="value_loss", simple_value=vl),
                    tf.Summary.Value(tag="entropy", simple_value=ent),
                    tf.Summary.Value(tag="kl", simple_value=kl),
                    tf.Summary.Value(tag="e_policy_loss", simple_value=e_pl),
                    tf.Summary.Value(tag="e_value_loss", simple_value=e_vl),
                    tf.Summary.Value(tag="e_entropy", simple_value=e_ent),
                    tf.Summary.Value(tag="e_kl", simple_value=e_kl),
                ],)
                self.file_writer.add_summary(stats, self.global_step)

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    # def _save(self, checkpoint_dir):
    def _save(self):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            # os.path.join(checkpoint_dir, "checkpoint"),
            os.path.join(self.logdir, "checkpoint"),
            global_step=self.global_step)
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_evaluators])
        extra_data = [
            self.local_evaluator.save(),
            self.global_step,
            self.kl_coeff,
            agent_state]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        self.global_step = extra_data[1]
        self.kl_coeff = extra_data[2]
        ray.get([
            a.restore.remote(o)
                for (a, o) in zip(self.remote_evaluators, extra_data[3])])

    def compute_action(self, observation):
        observation = self.local_evaluator.obs_filter(
            observation, update=False)
        return self.local_evaluator.common_policy.compute(observation)[0]
