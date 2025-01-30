import torch
import torch.nn as nn
from torch.distributions import Normal


class CNNActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        image_shape=[40,80],
        feature_dim=64,
        activation="relu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.feature_dim = feature_dim
        #feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.BathNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BathNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.feature_dim, kernel_size=3, stride=1, padding=0),
            nn.BathNorm2d(self.feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        """
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(image_shape).unsqueeze(0).unsqueeze(0)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
        )
        """

        self.num_additional_actor_obs = num_actor_obs - (image_shape[0] * image_shape[1])
        self.num_additional_critic_obs = num_critic_obs - (image_shape[0] * image_shape[1])

        mlp_input_dim_a = self.num_additional_actor_obs + feature_dim
        mlp_input_dim_c = self.num_additional_critic_obs + feature_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), bias=False)
        actor_layers.append(nn.BatchNorm1d(actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions, bias=True))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1], bias=False))
                actor_layers.append(nn.BatchNorm1d(actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0], bias=False))
        critic_layers.append(nn.BatchNorm1d(critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1, bias=True))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1], bias=False))
                critic_layers.append(nn.BatchNorm1d(critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.initialize_weights()

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def initialize_weights(self):
        """
        Initialize Model Weights
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    '''
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
    '''

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    # obervations is (N, I+x) where I is the image and x is the additional observations
    def feature_extractor(self, observations):
        flat_images = observations[:, :-self.num_additional_actor_obs]
        images = flat_images.view(-1, 1, 40, 80)
        features = self.cnn(images)
        ### features = self.linear(out)
        return features        
        

    def update_distribution(self, observations):
        features = self.feature_extractor(observations)
        obs = torch.cat((features, observations[:, -self.num_additional_actor_obs:]), dim=-1)
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        features = self.feature_extractor(observations)
        obs = torch.cat((features, observations[:, -self.num_additional_actor_obs:]), dim=-1)
        mean = self.actor(obs)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        features = self.feature_extractor(critic_observations)
        obs = torch.cat((features, critic_observations[:, -self.num_additional_critic_obs:]), dim=-1)
        value = self.critic(obs)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None