import os
import time
import torch
import statistics
import gymnasium as gym
import wandb # TODO: does this start wandb?
from collections import deque
from tqdm import tqdm
from typing import TYPE_CHECKING, Any

# from isaaclab.envs import ManagerBasedRLEnv # Ideally runner works for all env types

from .learner import MPAILLearner
from .utils import Stats

from mpail.mppi.core.vis import RolloutsVideo
if TYPE_CHECKING:
    from .mpail_cfg import MPAILRunnerCfg
    from wheeledlab_rl.configs import LogConfig

class MPAILRunner:

    def __init__(self,
        demonstrations: torch.Tensor,
        env: gym.Env,
        runner_cfg: 'MPAILRunnerCfg',
        log_cfg: 'LogConfig',
        device: str = "cuda"
    ):
        self.env = env
        self.cfg = runner_cfg
        self.log_cfg = log_cfg
        self.num_envs = env.unwrapped.num_envs
        self.device = device
        self.demonstrations = demonstrations.to(device)
        self.log_dir = log_cfg.run_log_dir

        # Create MPAIL Learner
        self.learner = MPAILLearner(self.demonstrations, self.num_envs,
                                    self.cfg.learner_cfg, device=device)

        # resolve dimensions of observations
        self._agent_state_dim = self.learner.state().shape[-1]
        self._num_actions = self.env.unwrapped.action_manager.total_action_dim

        self.learner.init_storage(
            self.num_envs,
            self.cfg.num_steps_per_env,
            [self._agent_state_dim],
            None, # No privileged observations
            [self._num_actions],
        )

        # Learning params
        self.num_steps_per_env = self.cfg.num_steps_per_env

        # Log
        match self.cfg.logger:
            case "wandb":
                # from rsl_rl.utils.wandb_utils import WandbSummarylogger
                # self.logger = WandbSummarylogger(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                import wandb
                self.logger = wandb
            case _:
                print("[INFO] No logger specified or not recognized.")
                self.logger = None

        if self.cfg.logger and self.cfg.vis_rollouts:
            assert self.learner.policy.vis is not None, "Policy must have visualization enabled for rollouts"
            self.rollouts_vid = RolloutsVideo(self.learner.policy.vis)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def train_mode(self):
        self.learner.policy.train()

    def learn(self):

        # start learning
        obs, infos = self.env.reset()
        critic_obs = obs
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + self.cfg.num_learning_iterations
        for it in tqdm(range(start_iter, tot_iter)):
            start = time.time()
            ep_stats = Stats()
            vis_ep = self.cfg.vis_rollouts and it % self.video_interval_its == 0
            # Rollout
            for _ in range(self.num_steps_per_env):

                with torch.inference_mode():

                    # Sample actions from policy
                    actions = self.learner.act(obs, vis_rollouts=vis_ep) # TODO: critic obs
                    # Step environment
                    next_obs, rewards, terms, truncs, infos = self.env.step(actions.to(self.env.unwrapped.device))

                    state = self.learner.state() # State with features

                    # Move to the agent device
                    dones = (terms | truncs).to(dtype=torch.long)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    # Process env step and store in buffer
                    stats = self.learner.process_env_step(rewards, dones, infos, next_obs)

                    if self.log_dir is not None:

                        # Log information
                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        # Update step stats
                        ep_stats.update(stats)

                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                self.learner.process_env_step_grad()
                if vis_ep:
                    self.rollouts_vid.update_video()

                # Update obs
                obs = next_obs

            with torch.inference_mode():

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.learner.compute_returns(state)

            # Log returns
            max_return = self.learner.storage.returns.max().cpu().numpy()
            mean_return = self.learner.storage.returns.mean().cpu().numpy()
            min_return = self.learner.storage.returns.min().cpu().numpy()

            # Update learner
            # Note: we keep arguments here since locals() loads them
            train_stats = self.learner.update(iteration=it) # Can return metrics for loss logging
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # TODO
            # Logging info and save checkpoint

            fps = int(self.num_steps_per_env * self.num_envs / (collection_time + learn_time))
            stats = {
                "Perf/collection_time": collection_time,
                "Perf/learn_time": learn_time,
                "Perf/fps": fps,
                "Env/mean_reward": statistics.mean(rewbuffer),
                "Env/mean_length": statistics.mean(lenbuffer),
                "Env/max_return": max_return,
                "Env/mean_return": mean_return,
                "Env/min_return": min_return,
                "it": it,
                "tot_iter": tot_iter,
            }
            train_stats.update(stats)
            train_stats.update(ep_stats.mean())

            if self.log_dir is not None and self.logger:

                self.logger.log(train_stats)

                # Log Rollout visualization
                if vis_ep:
                    vid_save_dir = os.path.join(self.log_dir, "rollouts_vis")
                    path_to_vid = self.rollouts_vid.save_video(
                        output_dir=vid_save_dir,
                        episode_num=it,
                        frame_rate=10
                    )
                    if not self.log_cfg.no_wandb:
                        wandb.log({"Rollouts Video": wandb.Video(path_to_vid)}, commit=False)
                        self.rollouts_vid.reset()

                # Save model
                if it % self.log_cfg.checkpoint_every == 0:
                    self.save(postfix=f"{it}")

        # Save the final model after training
        if self.log_dir is not None:
            self.save(postfix=f"{tot_iter}")

        if self.log_cfg.save_state_data and it % self.log_cfg.save_state_data_every == 0:
            # Save the state data
            os.makedirs(os.path.join(self.log_dir, "state_data"), exist_ok=True)
            state_data_path = os.path.join(self.log_dir, "state_data", f"state_data_{it}.pt")
            torch.save(self.learner.storage.observations, state_data_path)

    def save(self, postfix: str=""):

        saved_dict = {
            "model_state_dict": self.learner.policy.state_dict(),
            "disc_optimizer_state_dict": self.learner._disc_opt.state_dict(),
            "value_optimizer_state_dict": self.learner._disc_value_opt.state_dict(),
            "iter": self.current_learning_iteration,
        }

        model_path = os.path.join(self.log_dir, "models", f"model_{postfix}.pt")
        torch.save(saved_dict, model_path)

        # Save to external logger
        if self.logger:
            self.logger.save(model_path, base_path = os.path.dirname(model_path))


    def load(self, path: str, load_optimizer=True):
        ''' Load model from path '''

        saved_dict = torch.load(path, map_location=self.device)

        self.learner.policy.load_state_dict(saved_dict["model_state_dict"])
        if load_optimizer: # Does this point to the correct params?
            self.learner._disc_opt.load_state_dict(saved_dict["disc_optimizer_state_dict"])
            self.learner._disc_value_opt.load_state_dict(saved_dict["value_optimizer_state_dict"])

    @property
    def video_interval_its(self):
        return self.log_cfg.video_interval // self.num_steps_per_env