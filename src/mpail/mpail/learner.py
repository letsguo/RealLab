import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Dict

from mpail.mpail_cfg import ValueLearnerCfg
from mpail.cost import TDCost, GAIfOCost, CostToGo
from mpail.policy import MPAILPolicy

from .storage import MPAILRolloutStorage as Storage
from .utils import (
    compute_disc_train_stats, endless_dataloader, resolve_optim, GradAccumulator
)

if TYPE_CHECKING:
    from .mpail_cfg import  MPAILLearnerCfg


class MPAILLearner:
    '''Composes MPPIPolicy to include buffer population for MPAIL training as well as exploratory
    action selection.'''
    def __init__(self,
        demonstrations: torch.Tensor | Storage,
        num_envs: int,
        learner_config: 'MPAILLearnerCfg',
        device: str = "cuda",
        dtype=torch.float32
    ):

        self.cfg = learner_config
        self.disc_learner_cfg = self.cfg.disc_learner_cfg
        self.value_learner_cfg = self.cfg.value_learner_cfg

        self.policy = MPAILPolicy(self.cfg.policy_cfg, num_envs, device=device, dtype=dtype)
        self._costs:TDCost = self.policy.costs
        assert(hasattr(self._costs, "ss_cost"))
        assert(hasattr(self._costs, "ts_cost"))

        self.device = device
        self.dtype = dtype

        self.transition = Storage.Transition()

        # Learning params
        self._demonstrations = demonstrations # [T, traj_num, state_dim]
        self.num_mini_batches = self.cfg.num_mini_batches
        self.num_learning_epochs = self.cfg.num_learning_epochs

        #
        # DISCRIMINATOR SETUP
        #

        # Discriminator is logit of cost network (excludes final activation - LogSigmoid)
        self._disc:GAIfOCost.Discriminator = self._costs.ss_cost.get_discriminator()
        self._disc_opt = resolve_optim(self.disc_learner_cfg.opt)(
            self._disc.parameters(),
            **self.disc_learner_cfg.opt_params
        )

        #
        ## VALUE SETUP
        #

        self._disc_value:CostToGo.Value = self._costs.ts_cost.get_value()
        self._disc_value_opt = resolve_optim(self.value_learner_cfg.opt)(
            self._disc_value.parameters(),
            **self.value_learner_cfg.opt_params
        )

        #
        # ENTROPY SETUP
        #

        self._temp_entropy_grads = GradAccumulator([self.policy._temp_exp])
        self._temp_opt = resolve_optim("rmsprop")(
            [self.policy._temp_exp], lr=self.cfg.policy_cfg.temp_lr
        )

    def init_storage(self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # create rollout storage
        self.storage = Storage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            device=self.device,
        )

    def act(self, obs, critic_obs=None, vis_rollouts=False):
        '''Performs forward pass on MPPI and collects discriminator logits for MPAIL
        TODO: Critic obs?'''

        # Compute the actions and values
        self.transition.actions = self.policy.act(obs, vis_rollouts=vis_rollouts).detach()
        curr_state = self.state()

        if critic_obs is None:
            critic_obs = curr_state

        self.transition.values = self._disc_value(critic_obs).detach()

        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean().detach()
        self.transition.action_sigma = self.policy.action_stddev().detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = curr_state
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs) -> Dict[str, float]:
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts

        state = self.transition.observations.clone()
        next_state = self.policy.state()
        _obs_n_obs = torch.cat((state, next_state), dim=-1) # [num_envs, 2*state_dim]
        self.transition.rewards = self._costs.ss_cost.reward(_obs_n_obs).clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.value_learner_cfg.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Compute policy stats
        stats = self.policy.compute_stats()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

        return stats

    def process_env_step_grad(self):

        # Entropy target for temperature
        # TODO: is there a research question here by learning a policy through env-step loss?
        self._disc_opt.zero_grad()
        self._temp_opt.zero_grad()
        self._disc_value_opt.zero_grad()
        entropy = self.policy.entropy(recompute_weights=True) # Why is this computation unstable?
        loss = entropy.mean()
        loss.backward()

        # Save gradients
        self._temp_entropy_grads.add_grads()

    @property
    def _total_batch_size(self):
        return self.storage.num_transitions_per_env * self.storage.num_envs

    @property
    def _mini_batch_size(self):
        return self._total_batch_size // self.num_mini_batches

    def state(self):
        '''Returns the state of the MPPI policy'''
        return self.policy.state()

    def update(self, iteration: int) -> Dict[str, float]:

        mean_disc_value_loss = 0

        data_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # _demo_next_demo = torch.cat((
        #     self._demonstrations[:-1],
        #     self._demonstrations[1:]
        # ), dim=-1).flatten(0, 1) # [T * traj_num, 2*state_dim]
        _demo_next_demo = torch.cat(( # Yanda's data is [transitions, 2, state_dim]
            self._demonstrations[:, 0],
            self._demonstrations[:, 1]
        ), dim=-1) # [transitions, 2*state_dim]
        demo_batch_size = _demo_next_demo.shape[0] // self.num_mini_batches
        demo_generator = DataLoader(_demo_next_demo, batch_size=demo_batch_size, shuffle=True)
        train_stats  = {}

        train_disc_flag = (iteration % self.cfg.train_disc_every == 0)

        for ((
            state_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            next_obs_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch),
            demo_next_demo_batch
        ) in zip(data_generator, endless_dataloader(demo_generator)):

            # train critic
            mean_disc_value_loss += self.update_value(
                critic_obs_batch,
                target_values_batch,
                returns_batch,
                self._disc_value,
                self._disc_value_opt,
                self.value_learner_cfg,
            )

            # train discriminator
            if train_disc_flag:
                disc_train_stats = self.update_disc(
                    state_batch=state_batch,
                    next_state_batch=next_obs_batch,
                    demo_next_demo_batch=demo_next_demo_batch,
                )
                train_stats.update(disc_train_stats)

            # Update temperature
            self._temp_opt.zero_grad()
            for p, grad in self._temp_entropy_grads.get_grads().items():
                p.grad = grad # entropy target
            self._temp_opt.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_disc_value_loss /= num_updates
        train_stats.update({
            "Disc/mean_value_loss": mean_disc_value_loss.item(),
        })

        # -- Clear the storage
        self.storage.clear()

        # Clear grad accumulators
        self._temp_entropy_grads.reset()

        return train_stats

    def update_disc(self,
        state_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        demo_next_demo_batch: torch.Tensor,
    ):
        '''Trains discriminator network'''

        # stack s and s' for discriminator input
        _gen_input = torch.cat((state_batch, next_state_batch), dim=-1)

        self._disc_opt.zero_grad()

        _demo_logits = self._disc(demo_next_demo_batch)
        _gen_logits = self._disc(_gen_input)

        _input = torch.cat((
            _demo_logits,
            _gen_logits
        ))

        _target = torch.cat(( # [ 1 \\ 0 ]
            torch.ones_like(_demo_logits),
            torch.zeros_like(_gen_logits)
        ))
        _equalizing_weight = torch.cat((
            torch.ones_like(_demo_logits) * len(_gen_logits) / len(_demo_logits),
            torch.ones_like(_gen_logits)
        ))
        loss = F.binary_cross_entropy_with_logits(_input, _target, weight=_equalizing_weight)

        # add regularization loss
        loss += self.disc_learner_cfg.reg_coeff * sum(
            p.pow(2).sum() for p in self._disc.parameters()
        )

        # Renormalise the loss to be averaged over the whole
        # batch size instead of the minibatch size.
        loss /= self.cfg.num_mini_batches
        loss.backward()

        # do gradient step
        self._disc_opt.step()

        # Compute training stats
        with torch.no_grad():

            train_stats = compute_disc_train_stats(
                _input,
                _target,
                loss,
                from_probs=False
            )

            train_stats.update({
                "mean_demo_reward": _demo_logits.mean().item(),
                "mean_gen_reward": _gen_logits.mean().item(),
            })

        _train_stats = {}
        for key, value in train_stats.items(): # Add prefix to keys
            _train_stats[f"Disc/{key}"] = value

        return _train_stats

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self._disc_value(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.value_learner_cfg.gamma,
            self.value_learner_cfg.lam,
        )

    def update_value(self,
        critic_state_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        learner_cfg : ValueLearnerCfg,
    ):
        # train critic
        value_batch = model(critic_state_batch)
        if learner_cfg.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -learner_cfg.value_clip, learner_cfg.value_clip
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        loss = value_loss

        loss /= self.cfg.num_mini_batches

        # Gradient step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), learner_cfg.max_grad_norm
        )

        optimizer.step()

        return value_loss