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

    def forward(self, v_i, v_j, w_ij):
        u_i = self.embedding(v_i)

        if self.order == 2:
            u_j = self.context(v_j)
            # print(u_j)
            # print(u_i)
            # exit(0)
            negative = torch.sum(torch.exp(torch.matmul(u_j, u_i.T)), dim=0)
            # print(negative)
        else:
            u_j = self.embedding(v_j)

        inner_product = torch.sum(torch.mul(u_i, u_j), dim=1)
        # print(inner_product, torch.log(negative), inner_product - torch.log(negative))
        # print(torch.log(negative))

        if self.order == 2:
            loss = w_ij * (inner_product - torch.log(negative))
            # print(loss)
        else:
            loss = w_ij * F.logsigmoid(inner_product)
        # loss = w_ij * F.logsigmoid(inner_product)
        # print(loss)
        # exit(0)
        return -torch.mean(loss)

