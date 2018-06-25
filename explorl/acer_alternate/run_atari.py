#!/usr/bin/env python3
from baselines import logger
from explorl.acer_alternate.acer_simple import learn
from explorl.acer_alternate.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import atari_arg_parser #, make_atari_env

import datetime

# for collect env start =====================
from baselines.bench import Monitor
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import os
from env.collect import Collect
def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            # env = make_atari(env_id)
            env = Collect()
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env
        return _thunk
    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
# for collect env end =====================

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, logdir):
    env = make_atari_env(env_id, num_cpu, seed)
    evaluate_env = make_atari_env(env_id, 1, seed, wrapper_kwargs={'clip_rewards': False}, start_index=100)
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return
    learn(policy_fn, env, evaluate_env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
          replay_ratio=0, logdir=logdir)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    # parser.add_argument('--logdir', help ='Directory for logging')
    args = parser.parse_args()
    # logger.configure(args.logdir)
    logdir = './logs/'+datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    logger.configure(logdir)
    train(args.env, num_timesteps=1e8, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=2, logdir=logdir)
    # train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
    #       policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()
