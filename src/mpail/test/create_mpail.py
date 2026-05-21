###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Train an RL Agent in WheeledLab.")
parser.add_argument('-r', "--run-config-name", type=str, default="RSS_DRIFT_CONFIG", help="Run in headless mode.")
parser.add_argument('-e', '--expert-data-path', type=str, default=None, help="Path to expert data")
parser.add_argument('-v', '--video', type=bool, default=False, help="Record video during training")
simulation_app, args_cli = startup(parser=parser)

###################################

import os
import torch
import gymnasium as gym

from wheeledlab_tasks.navigation import MappingNavEnvCfg
from wheeledlab_rl.utils.custom_video_recorder import CustomRecordVideo
from wheeledlab_rl.utils import ClipAction

from mpail import MPAILRunner
from mpail.configs import MPAILRunnerConfig
from mpail.mppi.core.vis import RolloutsVideo

def main(
        num_envs: int = 64,
        demonstrations: torch.Tensor = None,
        device="cuda:0",
        log_dir="test/mpail/",
        num_steps: int = 1000,
        video = args_cli.video,
    ):

    env_cfg = MappingNavEnvCfg(num_envs=num_envs)
    env = gym.make("Isaac-MappingNavEnvRL-v0", cfg=env_cfg, render_mode="rgb_array")
    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env)

    # Wrap the environment in recorder
    if video:
        import wandb
        wandb.init(project="WheeledLab-research")
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % 2000 == 0,
            "video_length": 200,
            "disable_logger": True,
            "enable_wandb": True,
            "video_resolution": (1280, 720),
            "video_crf": 30,
        }
        print("[INFO] Recording videos during training.")
        env = CustomRecordVideo(env, **video_kwargs)

    # Create a new MPPI object
    runner_config = MPAILRunnerConfig()
    runner = MPAILRunner(demonstrations, env, runner_config, log_dir=log_dir, device=device)

    runner.learn()


if __name__ == "__main__":
    # load data from expert
    expert_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cleaned-expert-data.pt"
    )
    file = os.path.abspath(expert_data_path)
    demonstrations = torch.load(expert_data_path)
    main(demonstrations=demonstrations)
