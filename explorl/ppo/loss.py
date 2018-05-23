from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models import ModelCatalog
from explorl.models.visionnet import VisionNetwork


class ProximalPolicyLoss(object):

    other_output = ["vf_preds", "logprobs", "e_vf_preds", "e_logprobs", "curr_ent"]
    is_recurrent = False

    def __init__(
            self, observation_space, action_space,
            observations, value_targets, advantages, actions,
            prev_logits, prev_vf_preds,
            e_value_targets, e_advantages, e_prev_logits, e_prev_vf_preds,
            logit_dim, kl_coeff, distribution_class, config, sess):

        self.actions = actions
        self.value_targets = value_targets
        self.advantages = advantages
        self.prev_logits = prev_logits
        self.prev_vf_preds = prev_vf_preds
        self.e_value_targets = e_value_targets
        self.e_advantages = e_advantages
        self.e_prev_logits = e_prev_logits
        self.e_prev_vf_preds = e_prev_vf_preds

        # Saved so that we can compute actions given different observations
        self.observations = observations

        self.prev_dist = distribution_class(prev_logits)

        # for explore actor
        self.e_prev_dist = distribution_class(e_prev_logits)

        policy_net = VisionNetwork(observations, logit_dim, config["model"])
        self.curr_logits = policy_net.outputs
        self.curr_dist = distribution_class(self.curr_logits)
        self.sampler = self.curr_dist.sample()


        # for explore actor
        self.e_curr_logits = policy_net.last_layer
        self.e_curr_dist = distribution_class(self.e_curr_logits)
        self.e_sampler = self.e_curr_dist.sample()


        if config["use_gae"]:
            vf_config = config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            with tf.variable_scope("value_function"):
                self.value_net = VisionNetwork(observations, 1, vf_config)
                self.value_function = self.value_net.outputs


                # for explore value
                self.e_value_function = self.value_net.last_layer
                self.e_value_function = tf.reshape(self.e_value_function, [-1])
            self.value_function = tf.reshape(self.value_function, [-1])


        # Make loss functions.
        self.ratio = tf.exp(self.curr_dist.logp(actions) -
                            self.prev_dist.logp(actions))
        self.kl = self.prev_dist.kl(self.curr_dist)
        self.mean_kl = tf.reduce_mean(self.kl)
        self.entropy = self.curr_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.surr1 = self.ratio * advantages
        self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                      1 + config["clip_param"]) * advantages
        self.surr = tf.minimum(self.surr1, self.surr2)
        self.mean_policy_loss = tf.reduce_mean(-self.surr)


        # for explore actor loss
        self.e_ratio = tf.exp(self.e_curr_dist.logp(actions) -
                              self.e_prev_dist.logp(actions))
        self.e_kl = self.e_prev_dist.kl(self.e_curr_dist)
        self.e_mean_kl = tf.reduce_mean(self.e_kl)
        self.e_entropy = self.e_curr_dist.entropy()
        self.e_mean_entropy = tf.reduce_mean(self.e_entropy)
        self.e_surr1 = self.e_ratio * e_advantages
        self.e_surr2 = tf.clip_by_value(self.e_ratio, 1 - config["clip_param"],
                                      1 + config["clip_param"]) * e_advantages
        self.e_surr = tf.minimum(self.e_surr1, self.e_surr2)
        self.e_mean_policy_loss = tf.reduce_mean(-self.e_surr)


        if config["use_gae"]:
            # We use a huber loss here to be more robust against outliers,
            # which seem to occur when the rollouts get longer (the variance
            # scales superlinearly with the length of the rollout)
            self.vf_loss1 = tf.square(self.value_function - value_targets)
            vf_clipped = prev_vf_preds + tf.clip_by_value(
                self.value_function - prev_vf_preds,
                -config["clip_param"], config["clip_param"])
            self.vf_loss2 = tf.square(vf_clipped - value_targets)
            self.vf_loss = tf.minimum(self.vf_loss1, self.vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(self.vf_loss)
            self.loss = tf.reduce_mean(
                -self.surr + kl_coeff * self.kl +
                config["vf_loss_coeff"] * self.vf_loss -
                config["entropy_coeff"] * self.entropy)


            self.e_vf_loss1 = tf.square(self.e_value_function - e_value_targets)
            e_vf_clipped = e_prev_vf_preds + tf.clip_by_value(
                self.e_value_function - e_prev_vf_preds,
                -config["clip_param"], config["clip_param"])
            self.e_vf_loss2 = tf.square(e_vf_clipped - e_value_targets)
            self.e_vf_loss = tf.minimum(self.e_vf_loss1, self.e_vf_loss2)
            self.e_mean_vf_loss = tf.reduce_mean(self.e_vf_loss)
            self.e_loss = tf.reduce_mean(-self.e_surr + config["vf_loss_coeff"] * self.e_vf_loss)

            self.loss = self.loss + self.e_loss


        else:
            self.mean_vf_loss = tf.constant(0.0)
            self.loss = tf.reduce_mean(
                -self.surr +
                kl_coeff * self.kl -
                config["entropy_coeff"] * self.entropy)

        self.sess = sess

        if config["use_gae"]:
            # self.policy_results = [
            #     self.sampler, self.curr_logits, self.value_function]
            self.policy_results = [
                self.sampler, self.curr_logits, self.value_function,
                self.e_curr_logits, self.e_value_function,
                self.entropy]

            self.e_policy_results = [
                self.e_sampler, self.curr_logits, self.value_function,
                self.e_curr_logits, self.e_value_function,
                self.entropy]
        else:
            self.policy_results = [
                self.sampler, self.curr_logits, tf.constant("NA")]

    def compute(self, observation):
        # action, logprobs, vf = self.sess.run(
        #     self.policy_results,
        #     feed_dict={self.observations: [observation]})
        # return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0]}
        action, logprobs, vf, e_logprobs, e_vf, curr_ent = self.sess.run(
            self.policy_results,
            feed_dict={self.observations: [observation]})
        return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0],
                           "e_vf_preds": e_vf[0], "e_logprobs": e_logprobs[0],
                           "curr_ent": curr_ent[0]}

    def e_compute(self, observation):
        action, logprobs, vf, e_logprobs, e_vf, curr_ent = self.sess.run(
            self.e_policy_results,
            feed_dict={self.observations: [observation]})
        return action[0], {"vf_preds": vf[0], "logprobs": logprobs[0],
                           "e_vf_preds": e_vf[0], "e_logprobs": e_logprobs[0],
                           "curr_ent": curr_ent[0]}

    def loss(self):
        return self.loss

    def value(self, observation):
        vf, e_vf = self.sess.run(
            [self.value_function, self.e_value_function],
            feed_dict={self.observations: [observation]}
        )

        return vf[0], e_vf[0]


    def get_summary_data(self, feed_dict):
        feed_dict_in = {}
        for k, v in feed_dict.items():
            if hasattr(self, k):
                feed_dict_in[getattr(self, k)] = v

        return self.sess.run(
            [self.mean_policy_loss, self.mean_vf_loss, self.mean_entropy, self.mean_kl,
             self.e_mean_policy_loss, self.e_mean_vf_loss, self.e_mean_entropy, self.e_mean_kl,
            ],
            feed_dict=feed_dict_in)
        # return self.sess.run(
        #     [self.loss, self.policy_loss, self.value_loss, self.value_loss_1, self.kl_loss, self.entropy,
        #      self.spatial_actions, self.values, self.advantage],
        #     feed_dict=feed_dict_in)
