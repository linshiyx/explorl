#!/usr/bin/env python3
from baselines import logger
from explorl.acer_baseline.acer_simple import learn
from explorl.acer_baseline.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
import datetime

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, logdir):
    env = make_atari_env(env_id, num_cpu, seed)
    evaluate_env = make_atari_env(env_id, 1, seed, start_index=100)
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
    args = parser.parse_args()
    logdir = './logs/'+datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    logger.configure(logdir)
    train(args.env, num_timesteps=1e8, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=2, logdir=logdir)

if __name__ == '__main__':
    main()
