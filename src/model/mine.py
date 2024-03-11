import torch
import torch.nn as nn
import math
EPS = 1e-6


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, alpha=0.01):
        super(MINE, self).__init__()
        input_dim = x_dim + y_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.ELU()
        self.alpha = alpha
        self.running_mean = 0

    def forward(self, x, y):
        # (x;y|z)
        joint = torch.cat([x, y], dim=-1)
        marginal = torch.cat([x, y[torch.randperm(y.shape[0])]], dim=-1)

        # use a network to estimate the mutual information, so the same network to represent the MI()
        t_joint = self.network(joint).reshape(-1, 1)
        t_marginal = self.network(marginal).reshape(-1, 1)
        mi = torch.mean(t_joint) - (torch.logsumexp(t_marginal, 0).squeeze() - math.log(t_marginal.shape[0]))
        return mi

    def network(self, input):
        h1 = self.act(self.fc1(input))
        h2 = self.act(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3