"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from explorl.acer_alternate.acer_simple import learn
from explorl.acer_alternate.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines import logger
from baselines.common.vec_env import subproc_vec_env
from baselines.common import set_global_seeds
from retro_contest.local import make
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import datetime
import os
import random
import time

np.set_printoptions(precision=2, suppress=True)

is_remote = (os.getcwd() == "/root/compo")


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return self.action_key_mapping(ch)

    def action_key_mapping(self, key_action):
        map_action = 0

        if key_action == 'a':
            map_action = 0
        elif key_action == 'd':
            map_action = 1
        elif key_action == 's':
            map_action = 2
        elif key_action == 'f':
            map_action = 3
        elif key_action == 'u':
            exit()

        return map_action

getch = _GetchUnix()


def make_sonic_env(game, state, num_env, seed, scale_rew=True, start_index=0, render=False, wrapper_kwargs=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    # if wrapper_kwargs is None: wrapper_kwargs = {}
    def env_id_decorator(rank, scale_rew, render): # pylint: disable=C0111
        def _thunk():
            return make_env(game=game, state=state, seed=rank, scale_rew=scale_rew, render=render)
        return _thunk
    set_global_seeds(seed)
    return subproc_vec_env.SubprocVecEnv([env_id_decorator(i + start_index, scale_rew=scale_rew, render=render) for i in range(num_env)])


# def make_env(stack=True, scale_rew=True, game=None, state=None, seed=0, render=False):
def make_env(stack=True, scale_rew=True, game='SonicTheHedgehog-Genesis',
             state='SpringYardZone.Act1', seed=0, render=True):
    """
    Create an environment with some standard wrappers.
    """
    # if not is_remote:
    #     if game is None or state is None:
    #         import data_set_reader
    #         train_set = data_set_reader.read_train_set()
    #         game, state = random.choice(train_set)
    #     print("it's local env: ", game, state)
    #     from retro_contest.local import make
    #     env = make(game=game, state=state)
    # else:
    #     print("it's remote env")
    #     import gym_remote.client as grc
    #     env = grc.RemoteEnv('tmp/sock')
    env = make(game=game, state=state)
    env.seed(seed)
    env = AllowBacktracking(env)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(seed)), allow_early_resets=True)
    env = SonicDiscretizer(env, render)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, render):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
        #            ['DOWN', 'B'], ['B']]
        actions = [['LEFT'], ['RIGHT'], ['DOWN'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

        self.render = render

    def action(self, a): # pylint: disable=W0221
        if self.render:
            self.env.render()
            time.sleep(0.03)
        return self._actions[a].copy()

    # def step(self, action):
    #     key_action = getch()
    #     arr_action = self.action(key_action)
    #     obs, rew, done, info = self.env.step(arr_action)
    #     return obs, rew, done, info


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


def train(game, state, num_timesteps, seed, policy, lrschedule, num_cpu, logdir, load_info):
    # env = make_atari_env(env_id, num_cpu, seed)
    # evaluate_env = make_atari_env(env_id, 1, seed, wrapper_kwargs={'clip_rewards': False}, start_index=100)
    # env = subproc_vec_env.SubprocVecEnv([make_env()])
    set_global_seeds(seed)
    env = make_sonic_env(game=game, state=state, num_env=num_cpu, seed=0, render=False)
    evaluate_env = make_sonic_env(game=game, state=state, num_env=1, seed=1000, scale_rew=False)
    # for human play
    # env = DummyVecEnv([make_env])
    # evaluate_env = None
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return
    learn(policy_fn, env, evaluate_env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
          replay_ratio=0, logdir=logdir, load_info=load_info)
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
    # train(args.env, num_timesteps=1e8, seed=args.seed,
    #       policy=args.policy, lrschedule=args.lrschedule, num_cpu=16, logdir=logdir)
    game = 'SonicTheHedgehog-Genesis'
    state = 'SpringYardZone.Act1'
    # load_model_steps = 3833
    # load_model_rewards = 4948
    # load_path = "logs/0_628_998/{}_{}".format(load_model_steps, load_model_rewards)
    # load_info = {'path': load_path,
    #              'steps': load_model_steps,
    #              'rewards': load_model_rewards}
    load_model_steps = 4992
    load_model_rewards = 5163
    # load_model_steps = 8047
    # load_model_rewards = 6426
    # load_path = "logs/0_628_998_3833/{}_{}".format(load_model_steps, load_model_rewards)
    load_path = "logs/3833_4992/{}_{}".format(load_model_steps, load_model_rewards)
    load_info = {'path': load_path,
                 'steps': load_model_steps,
                 'rewards': load_model_rewards}
    # load_info=None
    train(game=game, state=state, num_timesteps=1e8, seed=args.seed, policy=args.policy,
          lrschedule=args.lrschedule, num_cpu=1, logdir=logdir, load_info=load_info)

if __name__ == '__main__':
    main()
