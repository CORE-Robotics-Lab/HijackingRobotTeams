import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class MLPPolicyNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, comm_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPPolicyNetwork, self).__init__()
        self.act_dim = out_dim - comm_dim
        self.comm_dim = comm_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_comm1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_comm2 = nn.Linear(hidden_dim, self.comm_dim)
        self.fc3 = nn.Linear(hidden_dim, self.act_dim)
        self.fc_comm3 = nn.Linear(1, 2)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h1_comm = self.nonlin(self.fc_comm1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h2_comm = self.nonlin(self.fc_comm2(h1_comm))
        h2_comm = h2_comm.unsqueeze(-1)
        out_act = self.out_fn(self.fc3(h2))
        out_comm = self.out_fn(self.fc_comm3(h2_comm))
        return out_act, out_comm

class SurrogatePolicyNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, comm_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SurrogatePolicyNetwork, self).__init__()
        self.act_dim = out_dim - comm_dim
        self.comm_dim = comm_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_comm1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_comm2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_comm3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_comm4 = nn.Linear(hidden_dim, self.comm_dim)

        self.fc5 = nn.Linear(hidden_dim, self.act_dim)
        self.fc_comm5 = nn.Linear(1, 2)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc5.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h1_comm = self.nonlin(self.fc_comm1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h2_comm = self.nonlin(self.fc_comm2(h1_comm))

        h3 = self.nonlin(self.fc3(h2))
        h3_comm = self.nonlin(self.fc_comm3(h2_comm))
        h4 = self.nonlin(self.fc4(h3))
        h4_comm = self.nonlin(self.fc_comm4(h3_comm))

        h4_comm = h4_comm.unsqueeze(-1)
        out_act = self.out_fn(self.fc5(h4))
        out_comm = self.out_fn(self.fc_comm5(h4_comm))
        return out_act, out_comm

class DiscrimNetwork(nn.Module):
    """
    MLP network (can be used as value or policy or discriminator)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, norm_in=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DiscrimNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.out_layer = nn.Softmax(dim=-1)

    def extract_reward(self, X):
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        rew = self.fc3(h2)
        return rew

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # print(X.shape)
        out = self.out_layer(self.extract_reward(X))
        return out

class MLPNetworkSoftMax(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, norm_in=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetworkSoftMax, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin_relu = nonlin
        self.nonlin_sig = nn.Sigmoid()
        # self.out_fn = lambda x: x


    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin_relu(self.fc1(self.in_fn(X)))
        h2 = self.nonlin_relu(self.fc2(h1))
        out = self.nonlin_sig(self.fc3(h2))
        return out


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers = 1):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # Nonlinear Layer
        self.nonlin = F.tanh
        # Embedding Layer
        self.fc1 = nn.Linear(input_size, 1)
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.out_layer = nn.Sigmoid()
    
    def forward(self, x): # x.shape: (batch_size, seq_len=1, feature_num)
        
        batch_size = x.size(0)

        embeddings = self.fc1(x) # embeddings.shape: (batch_size, seq_len=1)

        embeddings_expand = embeddings.t() # embeddings_expand.shape: (seq_len=1, batch_size)

        # # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(embeddings_expand, hidden) # out.shape: (seq_len=1, hidden_size); hidden.shape: (1,hidden_size)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.out_layer(self.fc2(out))

        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

class AIRLDiscrim(nn.Module):

    def __init__(self, obs_shape, acs_shape, gamma=0.9,
                hidden_dim=64,
                device ="cuda"):
        super().__init__()

        self.g = MLPNetwork(obs_shape+acs_shape, 1, hidden_dim=hidden_dim, constrain_out=False, norm_in=True).to(device)
        self.h = MLPNetwork(obs_shape, 1, hidden_dim=hidden_dim, constrain_out=False, norm_in=True).to(device)

        self.gamma = gamma

    def f(self, states, acs, dones, next_states):
        rs = self.g(torch.cat((states, acs), dim=-1)).squeeze()
        vs = self.h(states).squeeze()
        next_vs = self.h(next_states).squeeze()
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, acs, dones, log_pis, next_states):
        with torch.no_grad():
            # logits = self.forward(states, dones, log_pis, next_states)
            # return -F.logsigmoid(-logits)
            return self.g(torch.cat((states, acs), dim=-1)).squeeze()