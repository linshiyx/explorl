from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import os

from tensorflow.python import debug as tf_debug

import numpy as np

import ray
from ray.rllib.optimizers import PolicyEvaluator, SampleBatch
from ray.rllib.optimizers.multi_gpu_impl import LocalSyncParallelOptimizer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import _RLlibPreprocessorWrapper
# from ray.rllib.utils.sampler import SyncSampler
from explorl.utils.sampler import SyncSampler
# from ray.rllib.utils.filter import get_filter, MeanStdFilter
from explorl.utils.filter import get_filter, MeanStdFilter
# from ray.rllib.utils.process_rollout import process_rollout
from explorl.utils.process_rollout import process_rollout
# from ray.rllib.ppo.loss import ProximalPolicyLoss
from explorl.ppo.loss import ProximalPolicyLoss
from ray.rllib.models.preprocessors import AtariPixelPreprocessor

import gym

# TODO(rliaw): Move this onto LocalMultiGPUOptimizer
class PPOEvaluator(PolicyEvaluator):
    """
    Runner class that holds the simulator environment and the policy.

    Initializes the tensorflow graphs for both training and evaluation.
    One common policy graph is initialized on '/cpu:0' and holds all the shared
    network weights. When run as a remote agent, only this graph is used.
    """

    def __init__(self, env_id, config, logdir, is_remote):
        self.is_remote = is_remote
        if is_remote:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            devices = ["/cpu:0"]
        else:
            devices = config["devices"]
        self.devices = devices
        self.config = config
        self.logdir = logdir
        # self.env = ModelCatalog.get_preprocessor_as_wrapper(
        #     registry, env_creator(config["env_config"]), config["model"])
        env = gym.make(env_id)
        preprocessor = AtariPixelPreprocessor(env.observation_space, config["model"])
        self.env = _RLlibPreprocessorWrapper(env, preprocessor)
        if is_remote:
            config_proto = tf.ConfigProto()
        else:
            config_proto = tf.ConfigProto(**config["tf_session_args"])
        self.sess = tf.Session(config=config_proto)
        if config["tf_debug_inf_or_nan"] and not is_remote:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter(
                "has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Defines the training inputs:
        # The coefficient of the KL penalty.
        self.kl_coeff = tf.placeholder(
            name="newkl", shape=(), dtype=tf.float32)

        self.e_kl_coeff = tf.placeholder(
            name="e_newkl", shape=(), dtype=tf.float32)

        # The input observations.
        self.observations = tf.placeholder(
            tf.float32, shape=(None,) + self.env.observation_space.shape)
        # Targets of the value function.
        self.value_targets = tf.placeholder(tf.float32, shape=(None,))
        # Advantage values in the policy gradient estimator.
        self.advantages = tf.placeholder(tf.float32, shape=(None,))

        # for explore
        self.e_value_targets = tf.placeholder(tf.float32, shape=(None,))
        self.e_advantages = tf.placeholder(tf.float32, shape=(None,))

        action_space = self.env.action_space
        self.actions = ModelCatalog.get_action_placeholder(action_space)

        self.e_actions = ModelCatalog.get_action_placeholder(action_space)
        self.distribution_class, self.logit_dim = ModelCatalog.get_action_dist(
            action_space)
        # Log probabilities from the policy before the policy update.
        self.prev_logits = tf.placeholder(
            tf.float32, shape=(None, self.logit_dim))
        # Value function predictions before the policy update.
        self.prev_vf_preds = tf.placeholder(tf.float32, shape=(None,))

        # for explore
        self.e_prev_logits = tf.placeholder(
            tf.float32, shape=(None, self.logit_dim))
        self.e_prev_vf_preds = tf.placeholder(tf.float32, shape=(None,))

        if is_remote:
            self.batch_size = config["rollout_batchsize"]
            self.per_device_batch_size = config["rollout_batchsize"]
        else:
            self.batch_size = int(
                config["sgd_batchsize"] / len(devices)) * len(devices)
            assert self.batch_size % len(devices) == 0
            self.per_device_batch_size = int(self.batch_size / len(devices))

        def build_loss(obs, vtargets, advs, acts, plog, pvf_preds,
                       e_vtargets, e_advs, e_plog, e_pvf_preds):
            return ProximalPolicyLoss(
                self.env.observation_space, self.env.action_space,
                obs, vtargets, advs, acts, plog, pvf_preds,
                e_vtargets, e_advs, e_plog, e_pvf_preds,
                self.logit_dim,
                self.kl_coeff, self.distribution_class, self.config,
                self.sess)

        self.par_opt = LocalSyncParallelOptimizer(
            tf.train.AdamOptimizer(self.config["sgd_stepsize"]),
            self.devices,
            [self.observations, self.value_targets, self.advantages,
             self.actions, self.prev_logits, self.prev_vf_preds,
             self.e_value_targets, self.e_advantages, self.e_prev_logits, self.e_prev_vf_preds],
            self.per_device_batch_size,
            build_loss,
            self.logdir)

        # References to the model weights
        self.common_policy = self.par_opt.get_common_loss()
        self.variables = ray.experimental.TensorFlowVariables(
            self.common_policy.loss, self.sess)
        self.obs_filter = get_filter(
            config["observation_filter"], self.env.observation_space.shape)
        self.rew_filter = MeanStdFilter((), clip=5.0)
        self.e_rew_filter = MeanStdFilter((), clip=5.0)
        self.filters = {"obs_filter": self.obs_filter,
                        "rew_filter": self.rew_filter,
                        "e_rew_filter": self.e_rew_filter}
        self.sampler = SyncSampler(
            self.env, self.common_policy, self.obs_filter,
            self.config["horizon"], self.config["horizon"])
        self.init_op = tf.global_variables_initializer()
        # self.sess.run(tf.global_variables_initializer())

    def run_init_op(self):
        self.sess.run(self.init_op)

    def load_data(self, trajectories, full_trace):
        use_gae = self.config["use_gae"]
        dummy = np.zeros_like(trajectories["advantages"])
        return self.par_opt.load_data(
            self.sess,
            [trajectories["observations"],
             trajectories["value_targets"] if use_gae else dummy,
             trajectories["advantages"],
             trajectories["actions"],
             trajectories["logprobs"],
             trajectories["vf_preds"] if use_gae else dummy,
             trajectories["e_value_targets"] if use_gae else dummy,
             trajectories["e_advantages"],
             trajectories["e_logprobs"],
             trajectories["e_vf_preds"] if use_gae else dummy,
             ],
            full_trace=full_trace)

    def run_sgd_minibatch(
            self, batch_index, kl_coeff, full_trace, file_writer):
        return self.par_opt.optimize(
            self.sess,
            batch_index,
            extra_ops=[],
            # extra_ops=[
            #     self.mean_loss, self.mean_policy_loss, self.mean_vf_loss,
            #     self.mean_kl, self.mean_entropy],
            extra_feed_dict={self.kl_coeff: kl_coeff},
            file_writer=file_writer if full_trace else None)

    def compute_gradients(self, samples):
        raise NotImplementedError

    def apply_gradients(self, grads):
        raise NotImplementedError

    def save(self):
        filters = self.get_filters(flush_after=True)
        return pickle.dumps({"filters": filters})

    def restore(self, objs):
        objs = pickle.loads(objs)
        self.sync_filters(objs["filters"])

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def sample(self):
        """Returns experience samples from this Evaluator. Observation
        filter and reward filters are flushed here.

        Returns:
            SampleBatch: A columnar batch of experiences.
        """
        num_steps_so_far = 0
        all_samples = []

        while num_steps_so_far < self.config["min_steps_per_task"]:
            rollout = self.sampler.get_data()
            samples = process_rollout(
                rollout, self.rew_filter, self.e_rew_filter, self.config["gamma"],
                self.config["lambda"], use_gae=self.config["use_gae"])
            num_steps_so_far += samples.count
            all_samples.append(samples)
        return SampleBatch.concat_samples(all_samples)

    def sync_filters(self, new_filters):
        """Changes self's filter to given and rebases any accumulated delta.

        Args:
            new_filters (dict): Filters with new state to update local copy.
        """
        assert all(k in new_filters for k in self.filters)
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        """Returns a snapshot of filters.

        Args:
            flush_after (bool): Clears the filter buffer state.

        Returns:
            return_filters (dict): Dict for serializable filters
        """
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    def get_evaluate_metrics(self):

        policy_loss = []
        value_loss = []
        entropy = []
        kl = []
        e_policy_loss = []
        e_value_loss = []
        e_entropy = []
        e_kl = []
        reward = []
        length = []
        for i in range(self.config['num_evaluation']):
            rollout, rew, leng = self.sampler.get_episode_data()
            samples = process_rollout(
                rollout, self.rew_filter, self.e_rew_filter, self.config["gamma"],
                self.config["lambda"], use_gae=self.config["use_gae"])

            feed_dict = {
                'observations': samples["observations"],
                'value_targets': samples["value_targets"],
                'advantages': samples["advantages"],
                'actions': samples["actions"],
                # 'logprobs': samples["logprobs"],
                'prev_logits': samples["logprobs"],
                # 'vf_preds': samples["vf_preds"],
                'prev_vf_preds': samples["vf_preds"],
                'e_value_targets': samples["e_value_targets"],
                'e_advantages': samples["e_advantages"],
                # 'e_logprobs': samples["e_logprobs"],
                # 'e_vf_preds': samples["e_vf_preds"],
                'e_prev_logits': samples["e_logprobs"],
                'e_prev_vf_preds': samples["e_vf_preds"],
            }

            pl, vl, ent, k, e_pl, e_vl, e_ent, e_k = self.common_policy.get_summary_data(feed_dict)

            policy_loss.append(pl)
            value_loss.append(vl)
            entropy.append(ent)
            kl.append(k)
            e_policy_loss.append(e_pl)
            e_value_loss.append(e_vl)
            e_entropy.append(e_ent)
            e_kl.append(e_k)
            reward.append(rew)
            length.append(leng)

        return np.mean(policy_loss), np.mean(value_loss), np.mean(entropy), np.mean(kl), \
            np.mean(e_policy_loss), np.mean(e_value_loss), np.mean(e_entropy), np.mean(e_kl), \
            np.mean(reward), np.mean(length)
