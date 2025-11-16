import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.__init__() 

class ActorCritic(nn.Module):
    # MLP actor–critic: one network outputs the action probabilities, another outputs the scalar state-value.

    def __init__(self, state_dim, action_dim,
                 hidden: int = 64):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1))