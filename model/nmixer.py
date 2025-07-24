import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixerModel(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_shape: int,
        mixing_embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.mixing_embed_dim = mixing_embed_dim
        self.hypernet_layers = hypernet_layers
        self.hypernet_embed_dim = hypernet_embed_dim

        if hypernet_layers == 1:
            self.hyper_w1 = nn.Linear(
                self.state_shape, self.mixing_embed_dim * self.n_agents
            )
            self.hyper_w2 = nn.Linear(self.state_shape, self.mixing_embed_dim)

        elif hypernet_layers == 2:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim),
                nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.mixing_embed_dim * self.n_agents),
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim),
                nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.mixing_embed_dim),
            )

        else:
            raise ValueError("hypernet_layers must be 1 or 2")

        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_shape, self.mixing_embed_dim))
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs (torch.Tensor): [batch_size, episode_length-1, n_agents]
            states (torch.Tensor): [batch_size, episode_length-1, state_shape]
        Returns:
            torch.Tensor: [batch_size, episode_length-1, 1]
        """

        batch_size = agent_qs.shape[0]
        states = states.reshape(shape=(-1, self.state_shape))
        agent_qs = agent_qs.reshape(shape=(-1, 1, self.n_agents))

        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.reshape(shape=(-1, self.n_agents, self.mixing_embed_dim))
        b1 = self.hyper_b1(states)
        b1 = b1.reshape(shape=(-1, 1, self.mixing_embed_dim))

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.reshape(shape=(-1, self.mixing_embed_dim, 1))
        b2 = self.hyper_b2(states).reshape(shape=(-1, 1, 1))

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        y = torch.bmm(hidden, w2) + b2
        q_total = y.reshape(shape=(batch_size, -1, 1))

        return q_total
