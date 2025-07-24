# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAttention(nn.Module):
    def __init__(self, relation_num: int, hidden_dim: int, tar_hidden_dim: int):
        super().__init__()
        self.input_size = relation_num * hidden_dim
        self.relation_num = relation_num
        self.hidden_dim = hidden_dim
        self.tar_hidden_dim = tar_hidden_dim

        self.to_keys = nn.Linear(hidden_dim, tar_hidden_dim, bias=False)
        self.to_query = nn.Linear(hidden_dim * relation_num, tar_hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, agent_num, hidden_dim * relation_num) torch.Tensor
        """
        b, t, hin = x.size()
        assert (
            hin == self.input_size
        ), f"Input size {{hin}} should match {{self.input_size}}"

        query = self.to_query(x).view(b * t, self.tar_hidden_dim, 1)
        keys = F.tanh(self.to_keys(x.view(b * t, self.relation_num, self.hidden_dim)))

        e = self.tar_hidden_dim

        query = query / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.matmul(keys, query)
        dot = F.softmax(dot, dim=2)
        # print(dot.shape, x.shape)
        res = torch.matmul(
            x.view(b * t, self.relation_num, self.hidden_dim).transpose(1, 2), dot
        ).view(b, t, self.hidden_dim)
        return res
