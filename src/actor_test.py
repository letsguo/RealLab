import torch 
from actor_critic import ActorCritic

class PPOModel:
    def __init__(self, path):
        loaded_dict = torch.load(path, map_location=torch.device('cpu'))
        self.Model = ActorCritic(14,14,2, actor_hidden_dims=[128,128], critic_hidden_dims=[128,128])
        self.Model.load_state_dict(loaded_dict["model_state_dict"])
    def inference(self, state):
        return self.Model.act_inference(state)

if __name__ == "__main__":
    model = PPOModel("model_2000.pt")
    print(model.inference(-torch.ones(14)))
