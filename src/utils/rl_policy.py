import torch 
from utils.actor_critic import ActorCritic
import numpy

class RLModel:
    def __init__(self, name, acargs=(14,14,2), ackwargs = {'actor_hidden_dims': [128,128], 'critic_hidden_dims': [128,128]}):
        path = f"../models/{name}"
        loaded_dict = torch.load(path, map_location=torch.device('cpu'))
        self.Model = ActorCritic(*acargs, **ackwargs)
        self.Model.load_state_dict(loaded_dict["model_state_dict"])
    def inference(self, state):
        state = torch.Tensor(state).tolist()
        return self.Model.act_inference(state).tolist()
