import torch 
from utils.actor_critic import ActorCritic
import numpy

class RLModel:
    def __init__(self, name, acargs=(14,14,2), ackwargs = {'actor_hidden_dims': [128,128], 'critic_hidden_dims': [128,128]}):
        path = f"/root/catkin_ws/src/hound_core/src/models/{name}"
        loaded_dict = torch.load(path, map_location=torch.device('cpu'))
        self.Model = ActorCritic(*acargs, **ackwargs)
        self.Model.load_state_dict(loaded_dict["model_state_dict"])
    def inference(self, state):
        state = torch.Tensor(state)
        with torch.no_grad():
            actions = self.Model.act_inference(state).numpy().astype(numpy.float32)
            clipped_actions = numpy.clip(actions, -1, 1)
            return clipped_actions
