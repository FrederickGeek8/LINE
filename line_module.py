import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LINE(nn.Module):
    def __init__(self, size, latent_size=0, latent_dim=128, order=1):
        super(LINE, self).__init__()

        assert order in [1, 2], print("Proximity order must be 1 or 2.")

        self.order = order
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(size, latent_dim)
        self.embedding.weight.data = self.embedding.weight.data.uniform_(
            -0.5, 0.5) / latent_dim

        if self.order == 2:
            self.context = nn.Embedding(size, latent_dim)
            self.context.weight.data = self.context.weight.data.uniform_(
                -0.5, 0.5) / latent_dim

    def forward(self, v_i, v_j, neg_samples):
        u_i = self.embedding(v_i)

        if self.order == 2:
            u_j = self.context(v_j)
            negative = -self.context(neg_samples)
        else:
            u_j = self.embedding(v_j)
            negative = -self.embedding(neg_samples)

        # negative = torch.sum(torch.exp(torch.matmul(u_j, u_i.T)), dim=0)

        stage1 = F.logsigmoid(torch.sum(torch.mul(u_i, u_j), dim=1))

        stage2 = torch.mul(u_i.view(len(u_i), 1, self.latent_dim), negative)

        stage3 = torch.sum(F.logsigmoid(torch.sum(stage2, dim=2)), dim=1)

        return -torch.mean(stage1 + stage3)

