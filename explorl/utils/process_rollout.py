from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal
from ray.rllib.optimizers import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, reward_filter, e_reward_filter, gamma, lambda_=1.0, use_gae=True):
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (PartialRollout): Partial Rollout Object
        reward_filter (Filter): Filter for processing advantanges
        gamma (float): Parameter for GAE
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estamation

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards."""

    traj = {}
    trajsize = len(rollout.data["actions"])
    for key in rollout.data:
        traj[key] = np.stack(rollout.data[key])

    if use_gae:
        assert "vf_preds" in rollout.data, "Values not found!"
        vpred_t = np.stack(
            rollout.data["vf_preds"] + [np.array(rollout.last_r)]).squeeze()
        delta_t = traj["rewards"] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj["advantages"] = discount(delta_t, gamma * lambda_)
        traj["value_targets"] = traj["advantages"] + traj["vf_preds"]

        e_vpred_t = np.stack(
            rollout.data["e_vf_preds"] + [np.array(rollout.e_last_r)]).squeeze()
        for i in range(1, len(traj["curr_ent"])):
            traj['curr_ent'][i] = e_reward_filter(traj['curr_ent'][i])
        e_delta_t = np.concatenate([traj["curr_ent"][1:], np.array([0])]) + gamma * e_vpred_t[1:] - e_vpred_t[:-1]
        traj["e_advantages"] = discount(e_delta_t, gamma * lambda_)
        traj["e_value_targets"] = traj["e_advantages"] + traj["e_vf_preds"]
    else:
        rewards_plus_v = np.stack(
            rollout.data["rewards"] + [np.array(rollout.last_r)]).squeeze()
        traj["advantages"] = discount(rewards_plus_v, gamma)[:-1]

    for i in range(traj["advantages"].shape[0]):
        # print(type(traj["advantages"][i]))
        # print(traj["advantages"][i])
        traj["advantages"][i] = reward_filter(traj["advantages"][i])

        # traj["e_advantages"][i] = e_reward_filter(traj["e_advantages"][i])

    traj["advantages"] = traj["advantages"].copy()

    traj["e_advantages"] = traj["e_advantages"].copy()

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)
