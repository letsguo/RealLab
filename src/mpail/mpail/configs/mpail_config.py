# from isaaclab.utils import configclass
from typing import Callable, Optional
from dataclasses import dataclass, field

from .mppi_config import (
    MPPIConfig, DynamicsConfig, MapConfig, SamplingConfig, DynamicsNoActionConfig, VisConfig
)

from mpail.cost import GAIfOCost, CostToGo
from mpail.utils.costs import gail_cost_factory, value_cost_net_factory, fcnn_factory
from mpail import mpail_cfg

@dataclass(kw_only=True)
class GAIfOCostConfig(mpail_cfg.GAIfOCostCfg):

    state_dim: int = 16

    model_factory : Callable = fcnn_factory

    model_kwargs : dict = field(default_factory=lambda: {
        "hidden_dims": [16, 16],
        "activation": "lrelu",
        "use_spectral_norm": True,
        "squash_output": False,
    })


@dataclass(kw_only=True)
class CostToGoConfig(mpail_cfg.CostToGoCfg):

    model_factory : Callable = fcnn_factory

    model_kwargs : dict = field(default_factory=lambda:{
        "input_dim": 16, # TODO: get this from environment
        "hidden_dims": [32, 32],
        "activation": "relu",
    })

@dataclass(kw_only=True)
class MPAILCostConfig(mpail_cfg.TDCostCfg):

    single_step_class_cfg: GAIfOCostConfig = GAIfOCostConfig()

    terminal_state_class_cfg: CostToGoConfig = CostToGoConfig()

    ss_coeff: float = 1.

    ts_coeff: float = 1.

    risk: Optional[float] = None

@dataclass(kw_only=True)
class MPAILPolicyConfig(mpail_cfg.MPAILPolicyCfg):

    feat_dim: int = 4

    temperature: float = 1.0

    temp_lr: float = 1e-3

    min_temp: float = 1e-4

    opt_iters: int = 2

    u_per_command: int = 1

    debug: bool = False

    seed: int = 0

    # Learning parameters

    action_dist: str = "normal"

    action_dist_params: dict = field(default_factory=lambda: {
        "init_std": 1.0,
        "min_std": 0.01,
    })

    cost_cfg: mpail_cfg.MPAILTDCostCfg = MPAILCostConfig()

    dynamics_cfg: DynamicsConfig = DynamicsNoActionConfig(
        feat_dim = feat_dim,
        concatenate_feats = True,
    )

    map_cfg: MapConfig = MapConfig(
        feature_dim = feat_dim,
    )

    sampling_cfg: SamplingConfig = SamplingConfig(
        num_rollouts=1024,
        num_timesteps=20,
    )

    vis_cfg: VisConfig = None

@dataclass(kw_only=True)
class ValueLearnerConfig(mpail_cfg.ValueLearnerCfg):

    opt: str = "adam"
    '''Optimizer type'''

    opt_params: dict = field(default_factory=lambda: {
        "lr": 1e-4,
        "betas": (0.5, 0.999),
    })

    use_clipped_value_loss: bool = True
    '''Whether the value loss is clipped'''

    value_clip: float = 0.2
    '''If value loss is clipped, it's clipped to (-value_clip, value_clip)'''

    gamma: float = 0.99
    '''Discount factor'''

    lam: float = 0.95
    '''GAE lambda'''

    max_grad_norm: float = 1.0
    '''Clips the gradient norm of the value function parameters to this value'''

@dataclass(kw_only=True)
class DiscLearnerConfig(mpail_cfg.DiscLearnerCfg):

    opt: str = "adam"
    '''Optimizer type'''

    opt_params: dict = field(default_factory=lambda: {
        "lr": 1e-4,
        "betas": (0.5, 0.999),
    })

    reg_coeff: float = 1e-3


@dataclass(kw_only=True)
class MPAILLearnerConfig(mpail_cfg.MPAILLearnerCfg):

    num_mini_batches: int = 3
    '''Number of mini batches to train on'''

    num_learning_epochs: int = 3
    '''Number of epochs to train on'''

    train_disc_every: int = 3
    '''Trains discriminator every n iterations'''

    #
    # Discriminator
    #

    disc_learner_cfg: mpail_cfg.DiscLearnerCfg = DiscLearnerConfig()

    #
    # Value
    #

    value_learner_cfg: mpail_cfg.ValueLearnerCfg = ValueLearnerConfig()
    '''Configuration for the value learner'''

    #
    # Policy
    #

    policy_cfg: mpail_cfg.MPAILPolicyCfg = MPAILPolicyConfig()
    '''Configuration for the MPPI algorithm'''


@dataclass(kw_only=True)
class MPAILRunnerConfig(mpail_cfg.MPAILRunnerCfg):

    learner_cfg: mpail_cfg.MPAILLearnerCfg = MPAILLearnerConfig()

    num_learning_iterations: int = 100

    num_steps_per_env: int = 200

    logger: Optional[str] = "wandb"

    vis_rollouts: bool = True

    # Not used, but could maybe for training while debugging purposes?

    # enable_rl_value: bool = False

    # enable_imitation_value: bool = False

