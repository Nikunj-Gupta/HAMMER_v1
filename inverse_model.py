import argparse
import pickle

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cpu")

# We receive messages from the evaluation model of HAMMER.
# We also receive observations from a numpy array. (serving as ground )


class InverseModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(InverseModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


if __name__ == '__main__':

model = InverseModel(in_dim=1, out_dim=54)

# X = (import messages)
# Y = (import actual observations)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


n_epochs = 100 # or whatever
batch_size = 128 # or whatever

for epoch in range(n_epochs):

    # X is a torch Variable
    permutation = torch.randperm(X.size()[0])

    for i in range(0,X.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]

        # in case you wanted a semi-full example
        outputs = model.forward(batch_x)
        loss = F.mse_loss(outputs, batch_y)

        loss.backward()
        optimizer.step()


