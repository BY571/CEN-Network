import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim 
from typing import Tuple

class Policy(nn.Module):
    def __init__(self,observation_space: int, action_space: int, hidden_size: int=256, lr: float=1e-3):
        super(Policy, self).__init__()

        self.policy = nn.Sequential(nn.Linear(observation_space,hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, action_space)
                                )
        self.optimizer = optim.Adam(params=self.policy.parameters(), lr=lr)
    
    def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        x = self.policy(x)
        dist = Categorical(logits=x)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob
    
    def get_action(self, x: torch.Tensor):

        action, log_probs = self.forward(x)
        return action.detach().cpu().numpy(), log_probs
    
    def train(self, dones: torch.Tensor, rewards:torch.Tensor, log_probs: torch.Tensor)-> dict:
    
        disc_rewards = torch.cat(calc_discounted_rewards(rewards, dones)).unsqueeze(-1)
        baseline = disc_rewards.mean()
        loss = (-log_probs.unsqueeze(-1) * (disc_rewards-baseline).to(log_probs.device)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        log_dict = {"Loss": loss.item()}
        return log_dict

def calc_discounted_rewards(rewards: torch.Tensor, dones:torch.Tensor, gamma: float=0.99)-> torch.Tensor:
    R = 0
    discounted = []
  
    for idx in reversed(range(len(rewards))):
        R = rewards[idx] + R * gamma * dones[idx]
        discounted.insert(0, R.unsqueeze(-1))
    return discounted