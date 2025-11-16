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
        
    def act(self, state, memory):
        if state is None:
            state = np.zeros(1, dtype=np.float32)
        state = torch.tensor(state, dtype=torch.float32, device=device)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()
    

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy