import importlib

def load_policy(Config):
    """
    Load the policy class specified in the configuration.

    Args:
        Config (dict): Configuration dictionary.

    Returns:
        An instance of the specified policy class.
    """
    try:
        policy_module = Config["policy_config"]["policy_module"]
        policy_class = Config["policy_config"]["policy_class"]
        module = importlib.import_module(policy_module)
        policy_cls = getattr(module, policy_class)
        return policy_cls(Config)
    except (ImportError, AttributeError, KeyError) as e:
        raise ImportError(f"Failed to load policy {policy_class} from {policy_module}: {e}")
