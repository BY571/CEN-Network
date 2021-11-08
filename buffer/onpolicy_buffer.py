import numpy as np
import random
import torch
from collections import deque, namedtuple

class OnpolicyBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_list = []
        self.state_list = []
        self.next_state_list = []
        self.reward_list = []
        self.dones_list = []
        self.log_prob_list = []
    
    def add(self, state, action, reward, next_state, done, log_probs):
        """Add a new experience to memory."""
        self.action_list.append(torch.from_numpy(action).unsqueeze(0))
        self.state_list.append(torch.from_numpy(state).float().unsqueeze(0))
        self.next_state_list.append(torch.from_numpy(next_state).float().unsqueeze(0))
        self.reward_list.append(torch.from_numpy(np.array([reward])).float())
        self.dones_list.append(torch.from_numpy(np.array([1-done])).int())
        self.log_prob_list.append(log_probs.unsqueeze(0))
    
    def sample(self):
        states = torch.cat(self.state_list)
        self.state_list = []
        actions = torch.cat(self.action_list)
        self.action_list = []
        next_states = torch.cat(self.next_state_list)
        self.next_state_list = []
        rewards = torch.cat(self.reward_list)
        self.reward_list = []
        dones = torch.cat(self.dones_list)
        self.dones_list = []
        logprobs = torch.cat(self.log_prob_list)
        self.log_prob_list = []
        
        # print("s", states.shape,
        #       "ns", next_states.shape,
        #       "a", actions.shape,
        #       "r", rewards.shape,
        #       "d", dones.shape,
        #       "lp", logprobs.shape)
        
        return (states, actions, rewards, next_states, dones, logprobs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)