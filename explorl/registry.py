from ray.tune.registry import RLLIB_MODEL, _default_registry

def register_model(name, model):

    _default_registry.register(RLLIB_MODEL, name, model)
