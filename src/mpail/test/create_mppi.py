###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Train an RL Agent in WheeledLab.")
parser.add_argument('-c', "--config-dir", type=str, help="Load MPPI from torch state dict and yaml.")
simulation_app, args_cli = startup(parser=parser)

###################################

import torch
from tqdm import tqdm
import os

# Create Test Config
import gymnasium as gym

from wheeledlab_tasks.navigation import MappingNavEnvCfg

from mpail.configs import MPPIConfig, MPAILPolicyConfig
from mpail.mppi.core.vis import RolloutsVideo
from mpail.mppi import MPPIPolicy
from mpail import MPAILPolicy
from mpail.utils import dataclass_from_yaml_recurse, load_yaml

gym.register(
    id="Isaac-TestEnvRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MappingNavEnvCfg,
        "mppi_cfg_entry_point":MPPIConfig,
    }
)

def main(
        num_envs: int = 16,
        device="cuda:0",
        log_dir="test/mppi/",
        num_steps: int = 100,
        config_dir: str = None,
    ):

    if config_dir:
        # Load policy_config.yaml
        policy_config_path = os.path.join(config_dir, "policy_config.yaml")
        policy_config = load_yaml(policy_config_path)

        # Find the latest model_x.pt file
        model_files = [f for f in os.listdir(config_dir) if f.startswith("model_") and f.endswith(".pt")]
        latest_model_file = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        model_path = os.path.join(config_dir, latest_model_file)

        print(f"Loaded policy config from: {policy_config_path}")
        print(f"Loaded model from: {model_path}")

    env_cfg = MappingNavEnvCfg(num_envs=num_envs)
    env = gym.make("Isaac-TestEnvRL-v0", cfg=env_cfg)

    # Create a new MPPI object
    if config_dir:
        cfg_class = MPAILPolicyConfig
        mppi_config = dataclass_from_yaml_recurse(cfg_class, policy_config)  # Update MPPIConfig with loaded policy config
        mppi_policy = MPAILPolicy(
            num_envs=num_envs,
            policy_config=mppi_config,
            device=device,
        )
    else:
        cfg_class = MPPIConfig
        mppi_config = cfg_class()
        mppi_policy = MPPIPolicy(
            num_envs=num_envs,
            mppi_config=mppi_config,
            device=device,
        )

    if config_dir:
        policy_model = torch.load(model_path)['model_state_dict']
        mppi_policy.load_state_dict(policy_model)  # Load the model state dict

    vid = RolloutsVideo(mppi_policy.vis)

    obs, info = env.reset()
    for i in tqdm(range(num_steps)):
        action = mppi_policy.act(obs, vis_rollouts=True)
        obs, rew, term, trunc, info = env.step(action)
        vid.update_video()

    vid.save_video(output_dir=log_dir)

if __name__ == "__main__":
    main(config_dir=args_cli.config_dir)
