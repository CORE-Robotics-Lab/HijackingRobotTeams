import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import copy

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

# def onehot_from_logits(logits, eps=0.0):
#     """
#     Given batch of logits, return one-hot sample using epsilon greedy strategy
#     (based on given epsilon)
#     """
#     onehot_dim = logits.ndim - 1
#     # get best (according to current policy) actions in one-hot form
#     argmax_acs = (logits == logits.max(onehot_dim, keepdim=True)[0]).float()
#     if eps == 0.0:
#         return argmax_acs
#     # get random actions in one-hot form
#     rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
#         range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
#     # chooses between best and random actions using epsilon greedy
#     return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
#                         enumerate(torch.rand(logits.shape[0]))])

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    onehot_dim = logits.ndim - 1
    # get best (according to current policy) actions in one-hot form
    # argmax_acs = (logits == logits.max(onehot_dim, keepdim=True)[0]).float()
    # Find the index of the largest element along the last dimension
    largest_indices = torch.argmax(logits, dim=-1)

    # Create a new tensor with 1 at the largest indices and 0 elsewhere
    argmax_acs = torch.zeros_like(logits)
    argmax_acs.scatter_(dim=-1, index=largest_indices.unsqueeze(-1), value=1)
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])
                        
# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    # uniform_: Fills self tensor with numbers sampled from the continuous uniform distribution
    # https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html (Language: EN)
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

def two_hot_encode(explore, num_in_comm, num_in_move, action):
    device = "cuda" if action[0].is_cuda else "cpu"
    # bin = torch.Tensor([[0], [1]]).to(device)
    bin = torch.Tensor([0, 1]).to(device)
    if explore:
        if num_in_comm == 0 and num_in_move != 0:
            action = F.gumbel_softmax(action[0], hard=True, tau=1.0)
        elif num_in_comm != 0 and num_in_move == 0:
            action_comm_part = F.gumbel_softmax(action[1], hard=True, tau=1.0)
            action = action_comm_part @ bin
        else:
            action_move_part = F.gumbel_softmax(action[0], hard=True, tau=1.0)
            action_comm_part = F.gumbel_softmax(action[1], hard=True, tau=1.0)
            # print("action_comm_part = ", action_comm_part)
            # print("bin = ", bin)
            action_comm_part = action_comm_part @ bin
            # print("action_comm_part = ", action_comm_part)
            action = torch.cat([action_move_part, action_comm_part], dim=1)
    else:
        if num_in_comm == 0 and num_in_move != 0:
            action = onehot_from_logits(action[0])
        elif num_in_comm != 0 and num_in_move == 0:
            action_comm_part = onehot_from_logits(action[1])
            action = action_comm_part @ bin            
        else:
            action_move_part = onehot_from_logits(action[0])
            action_comm_part = onehot_from_logits(action[1]) # F.gumbel_softmax(action[1], hard=True, tau=1.0) 
            action_comm_part = action_comm_part @ bin
            action = torch.cat([action_move_part, action_comm_part], dim=1)
    return action

def two_hot_encode_action_only(explore, num_in_comm, num_in_move, action):
    device = "cuda" if action[0].is_cuda else "cpu"
    # bin = torch.Tensor([[0], [1]]).to(device)
    bin = torch.Tensor([0, 1]).to(device)
    if explore:
        if num_in_comm == 0 and num_in_move != 0:
            action = F.gumbel_softmax(action[0], hard=True, tau=1.0)
        elif num_in_comm != 0 and num_in_move == 0:
            action_comm_part = F.gumbel_softmax(action[1], hard=True, tau=1.0)
            action = action_comm_part @ bin
        else:
            action_move_part = F.gumbel_softmax(action[0], hard=True, tau=1.0)
            action_comm_part = F.gumbel_softmax(action[1], hard=True, tau=1.0)
            # print("action_comm_part = ", action_comm_part)
            # print("bin = ", bin)
            action_comm_part = action_comm_part @ bin
            # print("action_comm_part = ", action_comm_part)
            action = action_move_part
    else:
        if num_in_comm == 0 and num_in_move != 0:
            action = onehot_from_logits(action[0])
        elif num_in_comm != 0 and num_in_move == 0:
            action_comm_part = onehot_from_logits(action[1])
            action = action_comm_part @ bin            
        else:
            action_move_part = onehot_from_logits(action[0])
            action_comm_part = onehot_from_logits(action[1]) 
            action_comm_part = action_comm_part @ bin
            action = action_move_part
    return action

def soft_encode(num_in_comm, num_in_move, action):
    if num_in_comm == 0 and num_in_move != 0:
        action = F.softmax(action[0], dim=-1)
    elif num_in_comm != 0 and num_in_move == 0:
        action_comm_part = F.softmax(action[1], dim=-1)
        action = action_comm_part
    else:
        action_move_part = F.softmax(action[0], dim=-1)
        action_comm_part = F.softmax(action[1], dim=-1)
        # print("action_comm_part = ", action_comm_part)
        # print("bin = ", bin)
        # print("action_comm_part = ", action_comm_part)
        action = torch.cat([action_move_part, action_comm_part], dim=1)
    return action

def direct_encode(action):
    action_move_part = action[0]
    action_comm_part0 = action[1][...,0]
    action_comm_part1 = action[1][...,1]
    action = torch.cat([action_move_part, action_comm_part0, action_comm_part1], dim=1)
    return action

def act_in_act_soft_encode(action):
    action_move_part = action[0]

def flip_comm(actions, comm_agent_indices, motion_len, flip_vec, flip_num=8, mode="random"):
    flipped_num = 0
    comm_agent_num = len(comm_agent_indices)
    if mode == "random":
        for agent_i in comm_agent_indices:
            action_motion = actions[agent_i][:motion_len]
            action_comm = actions[agent_i][motion_len:]
            comm_len = len(action_comm)
            flip_indices = np.random.choice(comm_len, size=flip_num)
            actions[agent_i][motion_len:][flip_indices] = 1 - actions[agent_i][motion_len:][flip_indices]
    elif mode == "adv_comm":
        # comm_len = len(action_comm)
        flip_vec = np.split(flip_vec.detach().cpu().numpy(), indices_or_sections=comm_agent_num, axis=1)
        for agent_i in comm_agent_indices:
            # action_motion = actions[agent_i][:motion_len]
            # action_comm = actions[agent_i][motion_len:]
            flip_indices = np.where(flip_vec[agent_i].squeeze())
            actions[0][agent_i][motion_len:][flip_indices] = 1 - actions[0][agent_i][motion_len:][flip_indices]  
            flipped_num = flipped_num + len(flip_indices[0])
    elif mode == "direct":
        flip_vec = np.split(flip_vec.detach().cpu().numpy(), indices_or_sections=comm_agent_num, axis=1)
        for agent_i in comm_agent_indices:
            flipped_num = flipped_num + np.sum(actions[0][agent_i][motion_len:]*flip_vec[agent_i].squeeze())
            actions[0][agent_i][motion_len:] = flip_vec[agent_i].squeeze()
    return flipped_num

def update_obs_with_comm_vec(torch_obs, blind_agent_indices, flip_vec, mode="direct"):
    with torch.no_grad():
        torch_obs = copy.deepcopy(torch_obs)
        agg_comm_len = len(flip_vec[0])
        for agent_i, torch_obs_i in enumerate(torch_obs):
            if agent_i in blind_agent_indices:
                if mode == "direct":
                    flipped_num = torch.sum(torch_obs_i[0][-agg_comm_len:]!=flip_vec[0]).detach().cpu().numpy()
                    torch_obs_i[0][-agg_comm_len:] = flip_vec[0]
                elif mode == "flip":
                    flipped_num = torch.sum(flip_vec[0]).detach().cpu().numpy()
                    torch_obs_i[0][-agg_comm_len:] = flip_vec[0]*(1-torch_obs_i[0][-agg_comm_len:])+(1-flip_vec[0])*torch_obs_i[0][-agg_comm_len:]
    return torch_obs, flipped_num

def obs_to_comm_obs(torch_obs, obs_len_before_comm, blind_agent_indices=[3, 4], mode="append"):
    torch_obs = torch_obs.copy()
    torch_obs_comm = []
    for agent_i, torch_obs_i in enumerate(torch_obs):
        if agent_i not in blind_agent_indices:
            torch_obs_comm.append(torch_obs_i[0])
        elif agent_i in blind_agent_indices:
            if mode == "append":
                torch_obs_comm.append(torch_obs_i[0][:obs_len_before_comm])
            elif mode == "normal":
                torch_obs_comm.append(torch_obs_i[0])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    torch_obs_comm = torch.hstack(torch_obs_comm).unsqueeze(0)
    return torch_obs_comm

def obs_to_comms(torch_obs, len_combined_comm=32, blind_agent_indices=[3, 4]):
    torch_obs = torch_obs.copy()    
    any_blind_agent_idx = blind_agent_indices[0]
    comms = copy.deepcopy(torch_obs[any_blind_agent_idx][0][-len_combined_comm:])
    return comms

def get_agents_SA_pairs(agents_obs, agents_acs):
    agents_SA_pairs = []
    for agent_obs, agent_acs in zip(agents_obs, agents_acs):
        agents_SA_pairs.append(torch.cat((agent_obs, agent_acs), dim=-1))
    return agents_SA_pairs