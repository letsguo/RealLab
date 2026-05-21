import os
import yaml
import torch

from mpail.configs import MPAILPolicyConfig
from mpail.utils import dataclass_from_yaml_recurse
from mpail import MPAILPolicy

def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data

if __name__ == "__main__":
    # Example usage
    config_dir = "test_mppi_save"

    policy_config_path = os.path.join(config_dir, "policy_config.yaml")
    policy_config = load_yaml(policy_config_path)

    # Find the latest model_x.pt file
    model_files = [f for f in os.listdir(config_dir) if f.startswith("model_") and f.endswith(".pt")]
    latest_model_file = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    model_path = os.path.join(config_dir, latest_model_file)
    policy_model = torch.load(model_path)['model_state_dict']

    print(f"Loaded policy config from: {policy_config_path}")
    print(f"Loaded model from: {model_path}")

    cfg_class = MPAILPolicyConfig
    mppi_config = dataclass_from_yaml_recurse(MPAILPolicyConfig, policy_config)
    # create MPPI
    mppi_policy = MPAILPolicy(
        num_envs=1,
        policy_config=mppi_config,
        device="cuda",
    )
    # load model
    mppi_policy.load_state_dict(policy_model)  # Load the model state dict

    print("Model loaded successfully.")