from abc import ABC, abstractmethod
import torch
# import yaml
from BeamNGRL.control.UW_mppi.MPPI import MPPI

class HoundPolicyBase(ABC):
    def __init__(self, Config):
        
        self.Config = Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float

        self.Dynamics_config = Config.get("Dynamics_config", {})
        self.Sampling_config = Config.get("Sampling_config", {})
        self.default_max_thr = self.Sampling_config.get("max_thr", None)
        self.all_bad = False

    def set_hard_limit(self, hard_limit):
            
        self.Sampling_config["max_thr"] = min(
            hard_limit / self.Dynamics_config["throttle_to_wheelspeed"],
            self.default_max_thr,
        )

        self._update_policy_specific_hard_limit()

    def _update_policy_specific_hard_limit(self):
        """
        Hook method to be overridden by child classes if they need to perform
        additional updates when the hard limit is set.
        """
        pass
        
        
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the policy to its initial state. 
        To be implemented by all child classes.
        """
        pass