# from isaaclab.utils import configclass
from dataclasses import dataclass
from dataclasses import MISSING
from typing import Literal, Callable, Any, Optional
from torch.nn import Sequential
import os

from .mppi.core import (
    CostBaseCfg, NNCostCfg, MPPICfg
)
from .mppi.core.cost.utils import fcnn_factory

from .cost import TDCost, GAIfOCost, CostToGo

@dataclass(kw_only=True)
class GAIfOCostCfg(NNCostCfg):

    class_type: type[GAIfOCost] = GAIfOCost

    model_factory: Callable[[Any], Sequential] = fcnn_factory

    clip_costs: Optional[tuple[float, float]] = None

    state_dim: int = MISSING

    # feat_dim: int = MISSING # Backward compatibility; TODO: deprecate

    model_kwargs: dict = MISSING

@dataclass(kw_only=True)
class CostToGoCfg(NNCostCfg):

    class_type: type[CostToGo] = CostToGo

    clip_costs: Optional[tuple[float, float]] = None

    model_factory: Callable[[Any], Sequential] = fcnn_factory

@dataclass(kw_only=True)
class TDCostCfg(CostBaseCfg):

    single_step_class_cfg: CostBaseCfg = MISSING
    '''Config for the single step cost function'''

    terminal_state_class_cfg: CostBaseCfg = MISSING
    '''Config for the terminal state cost function'''

    ss_coeff: float = MISSING
    '''single step cost coefficient'''

    ts_coeff: float = MISSING
    '''terminal state cost coefficient for cost value function'''

    risk: Optional[float] = MISSING
    '''Discount factor for cost computation'''

    class_type: type[TDCost] = TDCost

@dataclass(kw_only=True)
class MPAILTDCostCfg(TDCostCfg):

    single_step_class_cfg: GAIfOCostCfg = MISSING
    '''Config for the single step cost function'''

    terminal_state_class_cfg: CostToGoCfg = MISSING
    '''Config for the terminal state cost function'''

    class_type: type[TDCost] = TDCost
    '''Type of cost function'''

    ss_coeff: float = 1.0

    ts_coeff: float = 1.0

@dataclass(kw_only=True)
class MPAILPolicyCfg(MPPICfg):

    action_dist: Literal["normal","categorical"] = MISSING
    '''Distribution of action selector'''

    action_dist_params: dict = MISSING
    '''Parameters for action distribution'''

    temp_lr: float = MISSING
    '''Learning rate for temperature'''

    cost_cfg: MPAILTDCostCfg = MISSING
    '''Configuration for sampling module'''

    min_temp: float = MISSING
    '''Minimum temperature for action distribution'''

@dataclass(kw_only=True)
class ValueLearnerCfg:

    opt: str = MISSING
    '''Optimizer type'''

    opt_params: dict = MISSING
    '''Optimizer parameters'''

    use_clipped_value_loss: bool = MISSING
    '''Whether the value loss is clipped'''

    value_clip: float = MISSING
    '''If value loss is clipped, it's clipped to (-value_clip, value_clip)'''

    gamma: float = MISSING
    '''Discount factor'''

    lam: float = MISSING
    '''GAE lambda'''

    max_grad_norm: float = MISSING
    '''Clips the gradient norm of the value function parameters to this value'''

@dataclass(kw_only=True)
class DiscLearnerCfg:

    opt: str = MISSING
    '''Optimizer type'''

    opt_params: dict = MISSING
    '''Optimizer parameters'''

    reg_coeff: float = MISSING
    '''Weight regularization coefficient for the discriminator'''

@dataclass(kw_only=True)
class MPAILLearnerCfg:

    num_mini_batches: int = MISSING
    '''Number of mini batches to train on'''

    num_learning_epochs: int = MISSING
    '''Number of epochs to train on'''

    train_disc_every: int = MISSING
    '''Trains discriminator every n iterations'''

    #
    # Disciminator
    #

    disc_learner_cfg : DiscLearnerCfg = MISSING
    '''Configuration for discriminator learning algorithm'''

    #
    # Value
    #

    value_learner_cfg : ValueLearnerCfg = MISSING
    '''Configuration for value function approximating discriminator logit returns'''

    #
    # Policy
    #

    policy_cfg: MPAILPolicyCfg = MISSING
    '''Configuration for MPPI module'''

@dataclass(kw_only=True)
class MPAILRunnerCfg:
    '''Manages stepping MPAIL and logging stats and training details'''

    learner_cfg: MPAILLearnerCfg = MISSING
    '''Configuration for MPPI'''

    num_steps_per_env: int = MISSING
    '''Number of steps per environment'''

    num_learning_iterations: int = MISSING
    '''Number of learning iterations'''

    path_to_demonstrations: str | os.PathLike = MISSING
    '''Number of steps per environment'''

    seed: int = MISSING
    '''Random seed'''

    logger: Optional[Literal["wandb"]] = MISSING
    '''Logger type'''

    enable_rl_value: bool = False
    '''Learn value function using rewards from rl environment'''

    enable_imitation_value: bool = False
    '''Learn value function using rewards from imitation'''

    save_separate_policy_cfg: bool = False
    '''Whether to save the policy config (.pkl) separately in another file'''
