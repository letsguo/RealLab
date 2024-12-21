# hound_actor_critic.py

from hound_policy import HoundPolicyBase
import torch
import numpy as np
from actor_critic import ActorCritic  # Ensure this is accessible in your project

class ActorCriticPolicy(HoundPolicyBase):
    def __init__(self, Config):
        super().__init__(Config)
        self.model_path = Config["ActorCritic_model_path"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # Load the model
        self.model = ActorCritic(
            num_actor_obs=14,
            num_critic_obs=14,
            num_actions=2,
            actor_hidden_dims=Config.get("actor_hidden_dims", [128, 128]),
            critic_hidden_dims=Config.get("critic_hidden_dims", [128, 128]),
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

        # Action limits (steering and throttle)
        self.max_steer = Config.get("max_steer", 0.488)  # Maximum steering angle in radians
        self.throttle_to_wheelspeed = Config["Dynamics_config"]["throttle_to_wheelspeed"]
        self.steering_max = Config["Dynamics_config"]["steering_max"]

    def update(self, state, *args, **kwargs):

        state_vector = np.copy(state[:14])

        state_tensor = torch.tensor(state_vector, device=self.device, dtype=self.dtype)

        # Perform inference
        with torch.no_grad():
            action_tensor = self.model.act_inference(state_tensor)

        # Convert action to numpy array
        action = action_tensor.cpu().numpy()

        # Clip actions to appropriate ranges
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], -1.0, 1.0)

        # Map throttle from [-1,1] to [0,1]
        throttle = (throttle + 1) / 2

        # Return the action in the expected format
        return np.array([steering, throttle])

    def reset(self):
        # If your model has any internal states, reset them here
        pass
