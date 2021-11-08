import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from typing import Tuple

class Ensemble_FC_Layer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super(Ensemble_FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, x) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class C_Network(nn.Module):
    def __init__(self,
                 observation_space: int,
                 action_space: int,
                 prediction_type: str="deterministic",
                 ensemble_size: int=7,
                 hidden_size: int=32,
                 output_size: int=6):
        super(C_Network, self).__init__()
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        self.layer1 = Ensemble_FC_Layer(observation_space + action_space, hidden_size, ensemble_size=ensemble_size)
        self.layer2 = Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size=ensemble_size)
        if prediction_type == "deterministic":
            self.layer3 = Ensemble_FC_Layer(hidden_size, output_size, ensemble_size)
        else:
            self.layer3 = Ensemble_FC_Layer(hidden_size, output_size * 2, ensemble_size)

    def forward(self, observation: torch.Tensor, action: torch.Tensor)-> torch.Tensor:

        inputs = torch.cat((observation, action), dim=-1)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1)
        x = torch.relu(self.layer1(inputs))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        
        
class N_Network(nn.Module):
    def __init__(self,
                 observation_space: int,
                 hidden_size: int=32,
                 prediction_type: str="deterministic", 
                 ensemble_size: int=7,
                 output_size: int=6):
        super(N_Network, self).__init__()
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        self.layer1 = Ensemble_FC_Layer(observation_space, hidden_size, ensemble_size)
        self.layer2 = Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size)
        if prediction_type == "deterministic":
            self.layer3 = Ensemble_FC_Layer(hidden_size, output_size, ensemble_size)
        else:
            self.layer3 = Ensemble_FC_Layer(hidden_size, output_size * 2, ensemble_size)
        
    def forward(self, observation: torch.Tensor)-> torch.Tensor:
        observation = observation[None, :, :].repeat(self.ensemble_size, 1, 1)
        x = torch.relu(self.layer1(observation))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class CENNetwork(nn.Module):
    def __init__(self,
                 observation_space: int,
                 action_space: int,
                 hidden_size: int=32,
                 prediction_type: str="deterministic",
                 ensemble_size: int=7, 
                 output_size: int=6,
                 lr: float=1e-4,
                 alpha: float=0.5):
        super(CENNetwork, self).__init__()
        self.action_space = action_space
        self.prediction_type = prediction_type
        self.output_size = output_size
        self.c_network = C_Network(observation_space,
                                   action_space,
                                   hidden_size=hidden_size,
                                   prediction_type=prediction_type,
                                   ensemble_size=ensemble_size,
                                   output_size=output_size)
        self.n_network = N_Network(observation_space,
                                   hidden_size=hidden_size,
                                   prediction_type=prediction_type,
                                   ensemble_size=ensemble_size,
                                   output_size=output_size)

        params = list(self.c_network.parameters()) + list(self.n_network.parameters())
        self.alpha = alpha
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=params, lr=lr)
    
    def forward(self, observation: torch.Tensor, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        h_a = self.c_network(observation, action)
        h_n = self.n_network(observation)
        
        # Do selection random/mean over ensemble
        # Use elite ensemble?
        if self.prediction_type == "deterministic":
            h_a = h_a.mean(0)
            h_n = h_n.mean(0)
        elif self.prediction_type == "probabilistic":
            h_a_means = h_a[:, :, :self.output_size].mean(0)
            h_a_std = (torch.clamp(h_a[:, :, self.output_size:].mean(0), -20, 2)).exp()

            h_n_means = h_n[:, :, :self.output_size].mean(0)
            h_n_std = (torch.clamp(h_n[:, :, self.output_size:].mean(0), -20, 2)).exp()
            
            h_a = torch.normal(h_a_means, h_a_std)
            h_n = torch.normal(h_n_means, h_n_std)
        else:
            raise ValueError
        return h_a, h_n
    
    def train(self, observation: torch.Tensor,
              action: torch.Tensor,
              next_observation: torch.Tensor)-> Tuple[torch.Tensor, float]:
        action = one_hot(action, self.action_space)
        delta_obs = next_observation - observation
        e_a, e_n = self.forward(observation, action)
        self.optimizer.zero_grad()
        loss = self.criterion(e_a + e_n, delta_obs) + self.alpha * self.criterion(e_n, delta_obs)
        loss.backward()
        self.optimizer.step()
        return e_a.detach(), {"CEN loss": loss.item()}