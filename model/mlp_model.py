import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers.mytrans import AttentionEncoder


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TransMLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.att = AttentionEncoder(
            n_layers=1, in_dim=hidden_size, hidden=hidden_size, dropout=0.1
        )
        self.fc3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        att = self.att(x)
        x = self.fc3(torch.concat([x, att], dim=-1))
        return x
