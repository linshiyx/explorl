import gym
import numpy as np

class Collect(gym.Env):

    low = 0
    high = 255
    observation_space = gym.spaces.Box(shape=(210, 160, 3), low=low, high=high)
    action_space = gym.spaces.Discrete(4)
    max_step = 500

    def seed(self, seed=None):
        pass

    def reset(self):
        self.num_step = 0

        self.pos = (150, 120)
        self.small_mineral_pos = (200, 150)
        self.big_mineral_pos = (50, 40)
        obs = np.ones(shape=self.observation_space.shape, dtype=np.uint32)
        obs[self.small_mineral_pos[0], self.small_mineral_pos[1], 1] = 0
        obs[self.big_mineral_pos[0], self.big_mineral_pos[1], 2] = 0
        obs[self.pos[0], self.pos[1], 0] = 0
        return obs

    def step(self, action=None):
        assert 0 <= action < 4
        self.num_step += 1

        if action == 0:
            self.pos = (max(self.pos[0]-1, 0), self.pos[1])
        elif action == 1:
            self.pos = (min(self.pos[0]+1, 20), self.pos[1])
        elif action == 2:
            self.pos = (self.pos[0], max(self.pos[1]-1, 0))
        elif action == 3:
            self.pos = (self.pos[0], min(self.pos[1]+1, 159))
        obs = np.ones(shape=self.observation_space.shape, dtype=np.uint32)
        obs[self.small_mineral_pos[0], self.small_mineral_pos[1], 1] = 0
        obs[self.big_mineral_pos[0], self.big_mineral_pos[1], 2] = 0
        obs[self.pos[0], self.pos[1], 0] = 0

        done = False
        reward = -0.01
        if self.pos == self.small_mineral_pos:
            reward = 50
            done = True
        elif self.pos == self.big_mineral_pos:
            reward = 1000
            done = True
        elif self.num_step >= self.max_step:
            done = True
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

