import ray
import gym
from ray.tune.registry import ENV_CREATOR, get_registry
from explorl.ppo.ppo import PPOAgent

env_id = "Alien-v4"

ray.init()

# registry = get_registry()
# env_creator = registry.get(ENV_CREATOR, env)
# env_creator = lambda env_config: gym.make(env)

ppo = PPOAgent(env_id)
ppo.run_init_op()
ppo.train()