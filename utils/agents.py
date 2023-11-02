from matplotlib.pyplot import axis
import numpy as np
from torch import Tensor
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.optim as optim
from .networks import MLPNetwork, MLPNetworkSoftMax, RNN, MLPPolicyNetwork, SurrogatePolicyNetwork, DiscrimNetwork, AIRLDiscrim
from .misc import soft_update, hard_update, average_gradients, two_hot_encode, two_hot_encode_action_only, soft_encode
from .noise import OUNoise
import copy

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')

class DDPGAgent(object):

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_in_comm, device="cuda", hidden_dim=64,
                 lr=0.01, discrete_action=True):

        self.policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                 comm_dim=num_in_comm,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 norm_in=True,
                                 discrete_action=discrete_action).to(device)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 norm_in=True).to(device)
        self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                        comm_dim=num_in_comm,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        norm_in=True,
                                        discrete_action=discrete_action).to(device)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        norm_in=True,).to(device)
        self.surrogate_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                 comm_dim=num_in_comm,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 norm_in=True,
                                 discrete_action=discrete_action).to(device)
        self.target_surrogate_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                 comm_dim=num_in_comm,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 norm_in=True,
                                 discrete_action=discrete_action).to(device)
        self.sharing_policy = MLPNetworkSoftMax(4, 1, hidden_dim=64).to(device)
        self.discriminator = DiscrimNetwork(input_dim=num_in_pol+num_out_pol, out_dim=2, hidden_dim=hidden_dim).to(device)
        self.reward_observer = MLPNetwork(num_in_pol+num_out_pol, 1, hidden_dim=hidden_dim, constrain_out=False, norm_in=True).to(device)
        self.airl_discriminator = AIRLDiscrim(num_in_pol, num_out_pol, gamma=0.9, hidden_dim=hidden_dim)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr) # original: lr
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.surrogate_policy_optimizer = Adam(self.surrogate_policy.parameters(), lr=10*lr)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=0.4*lr)
        self.airl_optimizer = Adam(self.airl_discriminator.parameters(), lr=lr)
        self.reward_optimizer = Adam(self.reward_observer.parameters(), lr=0.4*lr)
        self.sharing_policy_optimizer = Adam(self.sharing_policy.parameters(), lr=0.05*lr)

        self.scheduler_policy_optimizer = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.95)
        self.scheduler_critic_optimizer = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        self.scheduler_sharing_policy_optimizer = optim.lr_scheduler.StepLR(self.sharing_policy_optimizer, step_size=1000, gamma=0.95)
        self.scheduler_reward_optimizer = optim.lr_scheduler.StepLR(self.reward_optimizer, step_size=1000, gamma=0.95)
                                                            
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3 # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.num_in_move = num_out_pol - num_in_comm
        self.num_in_comm = num_in_comm
        if self.num_in_comm != 0: # normal predators
            self.obs_num_in_obs = num_in_pol
            self.comm_num_in_obs = 0
        else: # blind predators
            self.obs_num_in_obs = 22
            self.comm_num_in_obs = num_in_pol - self.obs_num_in_obs

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):

        action = self.policy(obs)
        if self.discrete_action:
            action = two_hot_encode(explore, self.num_in_comm, self.num_in_move, action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                # 'surrogate_policy': self.surrogate_policy.state_dict(),
                # 'target_surrogate_policy': self.target_surrogate_policy.state_dict(),
                'reward_observer': self.reward_observer.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        # hard_update(self.surrogate_policy, self.policy)
        # hard_update(self.target_surrogate_policy, self.surrogate_policy)

class AdvCommDDPGAgent(object):

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_in_comm, comm_agent_indices, gamma, tau, device="cuda", hidden_dim=64,
                 lr=0.01, discrete_action=True):

        self.policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                      comm_dim=num_in_comm,
                                      hidden_dim=hidden_dim,
                                      constrain_out=True,
                                      discrete_action=discrete_action).to(device)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False).to(device)
        self.target_policy = MLPPolicyNetwork(num_in_pol, num_out_pol,
                                              comm_dim=num_in_comm,
                                              hidden_dim=hidden_dim,
                                              constrain_out=True,
                                              discrete_action=discrete_action).to(device)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False).to(device)

        self.sharing_policy = MLPNetworkSoftMax(4, 1, hidden_dim=64).to(device)
        # self.sharing_policy = RNN(4, 1, hidden_dim=64).to(device)
                                        

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr) # original: lr
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.sharing_policy_optimizer = Adam(self.sharing_policy.parameters(), lr=0.05*lr)

        self.scheduler_policy_optimizer = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=1000, gamma=0.95)
        self.scheduler_critic_optimizer = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        self.scheduler_sharing_policy_optimizer = optim.lr_scheduler.StepLR(self.sharing_policy_optimizer, step_size=1000, gamma=0.95)
                                                            
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3 # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.num_in_move = num_out_pol - num_in_comm
        self.num_in_comm = num_in_comm
        self.comm_agent_indices = comm_agent_indices
        self.gamma = gamma
        self.device = device
        self.tau = tau
        self.niter = 0

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, torch_obs_comm, explore=False):

        action = self.policy(torch_obs_comm)

        if self.discrete_action:
            action = two_hot_encode(explore, self.num_in_comm, self.num_in_move, action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def update(self, sample, rl_ratio, mode="homo", parallel=False, logger=None, is_debug=True):

        if mode == "homo":
            obs, acs, rews, next_obs, dones, comms, next_comms= sample
            obs = obs[0]
            acs = acs[0]
            rews = rews[0]
            next_obs = next_obs[0]
            dones = dones[0]
            comms = comms[0]
            next_comms = next_comms[0]

            obs = copy.deepcopy(torch.cat((obs, comms), dim=-1))
            next_obs = copy.deepcopy(torch.cat((next_obs, next_comms), dim=-1))
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
            idxs = sample[1]
            rl_ratio = sample[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = sample # len(next_obs)=agent total num

        self.critic_optimizer.zero_grad()
        all_trgt_acs = two_hot_encode(explore=False, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=self.target_policy(next_obs))
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)
        target_value = (rews.view(-1, 1) + self.gamma * self.target_critic(trgt_vf_in) * (1 - dones.view(-1, 1)))
        vf_in = torch.cat((obs, comms), dim=1)
        actual_value = self.critic(vf_in)
        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        vf_loss = torch.mean(vf_loss_each.squeeze() * torch.Tensor(rl_ratio).to(device=self.device))
        vf_loss.backward()
        if parallel:
            average_gradients(self.critic)
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        curr_pol_out = self.policy(obs)
        curr_pol_vf_in = two_hot_encode(explore=True, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=curr_pol_out)
        vf_in = torch.cat((obs, curr_pol_vf_in), dim=1)
        pol_loss = -self.critic(vf_in).mean() + 0.05 * torch.sum(curr_pol_vf_in, dim=-1).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(self.policy)
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('comm_agent/losses',
                               {'vf_loss': torch.mean(vf_loss_each).item(),
                                'td_error': torch.mean(td_error_each).item(),
                                'pol_loss': pol_loss.item()},
                               self.niter)

        return td_error_abs_each

    def update_with_policy(self, sample, agent_i, maddpg, rl_ratio, recv_agents_indices=[3, 4], mode="homo", parallel=False, logger=None, is_debug=True):

        if mode == "homo":
            obs, acs, rews, next_obs, dones, comms, next_comms= sample
            obs = obs[0]
            acs = acs[0]
            rews = rews[0]
            next_obs = next_obs[0]
            dones = dones[0]
            comms = comms[0]
            next_comms = next_comms[0]
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
            idxs = sample[1]
            rl_ratio = sample[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = sample # len(next_obs)=agent total num
        agents_next_obs = list(torch.split(next_obs, split_size_or_sections=22, dim=1))
        agents_obs = list(torch.split(obs, split_size_or_sections=22, dim=1))
        agents_policies = maddpg.policies
        agents_target_policies = maddpg.target_policies
        self.critic_optimizer.zero_grad()
        # all_trgt_comm = torch.split(two_hot_encode(explore=False, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, 
        #                                           action=self.target_policy(next_obs)), 
        #                                           split_size_or_sections=int(self.num_in_comm//len(self.comm_agent_indices)), dim=1) 
        all_trgt_comm = two_hot_encode(explore=False, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=self.target_policy(torch.cat((next_obs, next_comms), dim=-1)))
        agents_next_obs_with_comm = self.reconstruct_obs_with_comm(agents_next_obs, all_trgt_comm, recv_agents_indices)
        all_trgt_acs = []
        for agent_i, agent_next_obs_with_comm in enumerate(agents_next_obs_with_comm):
            if agent_i in recv_agents_indices:
                all_trgt_acs.append(two_hot_encode(explore=False, num_in_comm=0, num_in_move=5, action=agents_target_policies[agent_i](agent_next_obs_with_comm)))
            else:
                all_trgt_acs.append(two_hot_encode(explore=False, num_in_comm=16, num_in_move=5, action=agents_target_policies[agent_i](agent_next_obs_with_comm)))
        trgt_vf_in = torch.cat((next_obs, *all_trgt_acs), dim=1)
        target_value = (rews.view(-1, 1) + self.gamma * self.target_critic(trgt_vf_in) * (1 - dones.view(-1, 1)))
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = self.critic(vf_in)
        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)
        vf_loss = torch.mean(vf_loss_each.squeeze() * torch.Tensor(rl_ratio).to(device=self.device))
        vf_loss.backward()
        if parallel:
            average_gradients(self.critic)
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        # curr_pol_vf_in = torch.split(two_hot_encode(explore=True, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=curr_pol_out), 
        #                                             split_size_or_sections=int(self.num_in_comm//len(self.comm_agent_indices)), dim=1)
        all_pol_comm = two_hot_encode(explore=True, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=self.policy(torch.cat((obs, comms), dim=-1)))
        agents_obs_with_comm = self.reconstruct_obs_with_comm(agents_obs, all_pol_comm, recv_agents_indices)
        all_pol_acs = []
        for agent_i, agent_obs_with_comm in enumerate(agents_obs_with_comm):
            if agent_i in recv_agents_indices:
                all_pol_acs.append(two_hot_encode(explore=True, num_in_comm=0, num_in_move=5, action=agents_policies[agent_i](agent_obs_with_comm)))
            else:
                all_pol_acs.append(two_hot_encode(explore=True, num_in_comm=16, num_in_move=5, action=agents_policies[agent_i](agent_obs_with_comm)))
        vf_in = torch.cat((obs, *all_pol_acs), dim=1)
        pol_loss = -self.critic(vf_in).mean() # + 10*torch.linalg.norm(all_pol_comm - comms, axis=-1).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(self.policy)
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('comm_agent/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

        return td_error_abs_each

    def zero_grad_all_agent_policies(self, agents_policies, agents_target_policies):
        for (policy, target_policy) in zip(agents_policies, agents_target_policies):
            policy.zero_grad()
            target_policy.zero_grad()
        return

    def remove_comm(self, agents_observations_or_actions, send_ag_num, recv_ag_num, obs_in_obs_len, ac_in_ac, copy_to_new, mode="obs"):
        if copy_to_new:
            agents_observations_or_actions = copy.deepcopy(agents_observations_or_actions)
        if mode == "obs":
            for agent_i, agent_obs in enumerate(agents_observations_or_actions):
                # INFO: We remove the comm part from the message-receiving agents' observations
                if not (agent_i < send_ag_num):
                    agents_observations_or_actions[agent_i] = agent_obs[:,:obs_in_obs_len]
                else:
                    pass
        elif mode == "ac":
            for agent_i, agent_ac in enumerate(agents_observations_or_actions):
                # INFO: We remove the comm part from the message-sending agents' actions
                if agent_i < send_ag_num:
                    agents_observations_or_actions[agent_i] = agent_ac[:,:ac_in_ac]
                else:
                    pass
        return agents_observations_or_actions

    def get_comm_from_acs(self, agents_acs, send_ag_num, acs_in_acs_len):
        comm_list = []
        for agent_i, agent_ac in enumerate(agents_acs):
            # INFO: We get the comm part from the message-sending agents' actions
            if (agent_i < send_ag_num):
                comm_list.append(agent_ac[:,acs_in_acs_len:])
            else:
                pass
        # INFO: We combine all agents' comm message together
        return torch.cat(comm_list, dim=-1)

    def get_agents_SA_pairs(self, agents_obs, agents_acs):
        agents_SA_pairs = []
        for agent_obs, agent_acs in zip(agents_obs, agents_acs):
            agents_SA_pairs.append(torch.cat((agent_obs, agent_acs), dim=-1))
        return agents_SA_pairs

    def get_discriminator_reward(self, obs, acs, agents_SA_pairs, maddpg, recv_agents_indices):
        discriminator_reward = 0
        for agent_i, agent_SA_pair in enumerate(agents_SA_pairs):
            if agent_i in recv_agents_indices:
                rew = maddpg.reward_observers[agent_i](agent_SA_pair)
                discriminator_res = torch.exp(rew) / (torch.exp(rew) + soft_encode(num_in_comm=maddpg.agents[agent_i].num_in_comm, num_in_move=maddpg.agents[agent_i].num_in_move, action=maddpg.agents[agent_i].surrogate_policy(obs[agent_i]))[acs[agent_i]==1].unsqueeze(1).detach()) 
                discriminator_reward = discriminator_reward + (torch.log(discriminator_res)).mean() + (-torch.log(1 - discriminator_res)).mean()
        return discriminator_reward

    def get_optimized_reward(self, obs, acs, agents_SA_pairs, maddpg, recv_agents_indices):
        # INFO: Estimate the reward with -log(\pi_surr(a))
        discriminator_reward = 0
        for agent_i, agent_SA_pair in enumerate(agents_SA_pairs):
            if agent_i in recv_agents_indices:
                current_agent = maddpg.agents[agent_i]
                discriminator_reward = discriminator_reward + (-torch.log(soft_encode(num_in_comm=current_agent.num_in_comm, num_in_move=current_agent.num_in_move, action=current_agent.surrogate_policy(obs[agent_i]))[acs[agent_i]==1]))
        return discriminator_reward / len(recv_agents_indices)

    def get_airl_reward(self, obs, acs, dones, next_obs, maddpg, recv_agents_indices):
        airl_reward = 0
        for agent_i in recv_agents_indices:
            current_agent = maddpg.agents[agent_i]
            log_pis = torch.log(soft_encode(num_in_comm=current_agent.num_in_comm, num_in_move=current_agent.num_in_move, action=current_agent.policy(obs[agent_i]))[acs[agent_i]==1])
            airl_reward = airl_reward + (-current_agent.airl_discriminator.calculate_reward(obs[agent_i], acs[agent_i], dones, log_pis, next_obs[agent_i]))
        return airl_reward

    def correct_with_updated_ag_pol(self, agents_obs, agents_next_obs, agents_target_policies, comm_agents_indices=[0], recv_agents_indices=[1, 2, 3], correct=True):
            # actions_with_comms = []
            comms = []
            if correct:
                for agent_i, agent_obs_with_comm in enumerate(agents_obs):
                    if agent_i in comm_agents_indices:
                        action_with_comm = two_hot_encode(explore=False, num_in_comm=16, num_in_move=5, action=agents_target_policies[agent_i](agent_obs_with_comm))
                        comms.append(action_with_comm[:,-16:])
                comms = torch.cat((comms), dim=-1)
                for agent_i, agent_next_obs_with_comm in enumerate(agents_next_obs):
                    if agent_i in recv_agents_indices:
                        agents_next_obs[agent_i][:,-48:] = comms
            else:
                pass
            return agents_next_obs, comms

    def update_with_offpolicy(self, sample, agent_i, maddpg, rl_ratio, flip_mode, max_flipping_num=8, recv_agents_indices=[3, 4], mode="homo", parallel=False, logger=None, is_debug=True):

        if mode == "homo":
            obs, acs, rews, next_obs, dones, comms, next_comms= sample
            obs = obs[0]
            acs = acs[0]
            rews = rews[0]
            next_obs = next_obs[0]
            dones = dones[0]
            comms = comms[0]
            next_comms = next_comms[0]
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
            idxs = sample[1]
            rl_ratio = sample[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = sample # len(next_obs)=agent total num

                
        agent_obs_with_comm_list = []

        # INFO: In the PCP env, no-comm-receiving agents (3 predators) have obs with size of 22 and comm-receiving agents (2 capturers) have obs with the size 22 + 16 * 3
        agents_next_obs = list(torch.split(next_obs, split_size_or_sections=[22, 22, 22, 70, 70], dim=1))
        agents_obs = list(torch.split(obs, split_size_or_sections=[22, 22, 22, 70, 70], dim=1))
        # INFO: In the PCP env, comm-sending agents (3 predators) have act with size of 5 + 16 and silent agents (2 capturers) have act with the size 5
        agents_acs = list(torch.split(acs, split_size_or_sections=[21, 21, 21, 5, 5], dim=1))
        # agents_dones = list(torch.split(dones, split_size_or_sections=[1, 1, 1, 1, 1], dim=1))
        agents_next_obs_only = self.remove_comm(agents_next_obs, send_ag_num=3, recv_ag_num=2, obs_in_obs_len=22, ac_in_ac=5, copy_to_new=True, mode="obs")
        agents_obs_only = self.remove_comm(agents_obs, send_ag_num=3, recv_ag_num=2, obs_in_obs_len=22, ac_in_ac=5, copy_to_new=True, mode="obs") 
        agents_acs_only = self.remove_comm(agents_acs, send_ag_num=3, recv_ag_num=2, obs_in_obs_len=22, ac_in_ac=5, copy_to_new=True, mode="ac") 
        comms = self.get_comm_from_acs(agents_acs, send_ag_num=3, acs_in_acs_len=5)
        agents_SA_pairs = self.get_agents_SA_pairs(agents_obs, agents_acs)
        # INFO: Estimate the reward
        rews_from_discriminator = self.get_optimized_reward(agents_obs, agents_acs, agents_SA_pairs, maddpg, recv_agents_indices).detach()

        # INFO: We assume we do NOT have the access to the ground truth policy so we use surrogate policy as the agent policy
        agents_policies = copy.deepcopy(maddpg.surrogate_policies) 
        agents_target_policies = copy.deepcopy(maddpg.target_surrogate_policies)  
        self.zero_grad_all_agent_policies(agents_policies, agents_target_policies)
        self.critic_optimizer.zero_grad()

        # INFO: Generate the adversarial message based on observation
        all_trgt_comm = two_hot_encode(explore=False, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=self.target_policy(next_obs))
        # INFO: Imagine that we are using adv message?
        agents_next_obs_with_comm = self.replace_obs_with_comm(agents_next_obs, all_trgt_comm, recv_agents_indices, mode=flip_mode)
        all_trgt_acs_only = []
        # INFO: Imagine what actions agents will take if they are using adversarial communication message? We can do that based on surr policy.
        for agent_i, agent_next_obs_with_comm in enumerate(agents_next_obs_with_comm):
            if agent_i in recv_agents_indices:
                all_trgt_acs_only.append(two_hot_encode_action_only(explore=False, num_in_comm=0, num_in_move=5, action=agents_target_policies[agent_i](agent_next_obs_with_comm)))
            else:
                all_trgt_acs_only.append(two_hot_encode_action_only(explore=False, num_in_comm=16, num_in_move=5, action=agents_target_policies[agent_i](agent_next_obs_with_comm)))
        # INFO: We train the adversarial Q based on <obs\comm, act\comm, nobs\comm, est. adv_rew> -- TD error
        trgt_vf_in = torch.cat((*agents_next_obs_only, *all_trgt_acs_only), dim=1)
        target_value = (rews_from_discriminator.view(-1, 1) + self.gamma * self.target_critic(trgt_vf_in) * (1 - dones.view(-1, 1)))

        vf_in = torch.cat((*agents_obs_only, *agents_acs_only), dim=1)
        actual_value = self.critic(vf_in)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        vf_loss = torch.mean(vf_loss_each.squeeze() * torch.Tensor(rl_ratio).to(device=self.device))
        vf_loss.backward()

        if parallel:
            average_gradients(self.critic)
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        # INFO: Decide if we correct the communication with the most updated surrogate policy. No need to do so (correct=False) for fixed-deployed agent policy.
        agents_next_obs_with_corrected_comm, next_comms_corrected = self.correct_with_updated_ag_pol(agents_obs, agents_next_obs, agents_target_policies, recv_agents_indices=[3, 4], correct=False)
        # INFO: Generate the adversarial message based on observation
        all_next_pol_corrected_modified_comm = two_hot_encode(explore=True, num_in_comm=self.num_in_comm, num_in_move=self.num_in_move, action=self.policy(torch.cat(agents_next_obs_with_corrected_comm, dim=-1)))
        # INFO: Imagine that we are using adv message?
        agents_next_obs_with_comm = self.replace_obs_with_comm(agents_next_obs_with_corrected_comm, all_next_pol_corrected_modified_comm, recv_agents_indices, mode=flip_mode)
        all_next_pol_acs_only = []
        # INFO: Imagine what actions agents will take if they are using adversarial communication message? Do some exploration here to cover more adv message.
        for agent_i, agent_next_obs_with_comm in enumerate(agents_next_obs_with_comm):
            if agent_i in recv_agents_indices:
                all_next_pol_acs_only.append(two_hot_encode_action_only(explore=True, num_in_comm=0, num_in_move=5, action=agents_policies[agent_i](agent_next_obs_with_comm)))
            else:
                all_next_pol_acs_only.append(two_hot_encode_action_only(explore=True, num_in_comm=16, num_in_move=5, action=agents_policies[agent_i](agent_next_obs_with_comm)))
            agent_obs_with_comm_list.append(agent_next_obs_with_comm.detach())
        vf_in = torch.cat((*agents_next_obs_only, *all_next_pol_acs_only), dim=1)
        
        # INFO: vf_in contains actions based on adv messages, therefore the adv policy can be trained if we backpropagate through self.critic
        if flip_mode == "flip":
            # INFO: Add L1 filipping number regularization term
            pol_loss = -self.critic(vf_in).mean() + 0.04 * np.maximum((self.niter - 3000) / 20000, 0) * all_next_pol_corrected_modified_comm.sum(dim=1).mean()
        elif flip_mode == "direct":
            flipping_num = torch.sum(abs(all_next_pol_corrected_modified_comm - comms), axis=-1)
            pol_loss = -self.critic(vf_in).mean() + 0.1 * np.maximum((self.niter - 3000) / 20000, 0) * flipping_num.mean()

        pol_loss.backward()
        if parallel:
            average_gradients(self.policy)
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        # INFO: Update the adversarial policy!
        self.policy_optimizer.step()
        self.zero_grad_all_agent_policies(agents_policies, agents_target_policies)
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('comm_agent/losses',
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)
            logger.add_scalars('comm_agent/flip_num_in_training', {'flipping_num': all_next_pol_corrected_modified_comm.sum(dim=1).mean()}, self.niter)

        return agent_obs_with_comm_list
    
    def reconstruct_obs_with_comm(self, agents_obs, all_trgt_comm, recv_agents_indices):
        only_comm_agent_idx = 0
        for agent_i, agent_obs in enumerate(agents_obs):
            if agent_i in recv_agents_indices:
                agents_obs[agent_i] = torch.concat([agent_obs, all_trgt_comm], dim=-1)
                only_comm_agent_idx = only_comm_agent_idx + 1
        return agents_obs
    
    def replace_obs_with_comm(self, agents_obs, all_trgt_comm, recv_agents_indices, mode="direct"):
            agents_obs = copy.deepcopy(agents_obs)
            recv_obs = []
            combined_comm_len = all_trgt_comm.shape[1]
            only_comm_agent_idx = 0
            for agent_i, agent_obs in enumerate(agents_obs):
                if agent_i in recv_agents_indices:
                    # INFO: Replace the communication in agent observation with the adversarial one such that the gradient can flow back
                    if mode == "direct":
                        agents_obs[agent_i][:,-combined_comm_len:] = all_trgt_comm
                        recv_obs.append(agents_obs[agent_i])
                    elif mode == "flip":
                        obs_in_obs_len = agents_obs[agent_i].shape[1] - combined_comm_len
                        ori_comm = agents_obs[agent_i][:,-combined_comm_len:] # original communication
                        # INFO: Adv Message = Original Message XOR adversary policy output -- differentiable!
                        recv_obs.append(torch.cat((agents_obs[agent_i][:,:obs_in_obs_len], all_trgt_comm*(1-ori_comm)+(1-all_trgt_comm)*ori_comm), dim=-1))
                    else:
                        NotImplementedError
                    only_comm_agent_idx = only_comm_agent_idx + 1
                else:
                    recv_obs.append(agents_obs[agent_i])
            return recv_obs

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def prep_training(self):
        self.policy.train()
        self.critic.train()
        self.target_policy.train()
        self.target_critic.train()

    def prep_rollouts(self, device):
        self.policy = self.policy.to(device)
        self.critic = self.critic.to(device)
        self.target_policy = self.target_policy.to(device)
        self.target_critic = self.target_critic.to(device)

        self.policy.eval()
        self.critic.eval()
        self.target_policy.eval()
        self.target_critic.eval()

    def update_comm_targets(self):

        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_policy, self.policy, self.tau)
        self.niter += 1

    def save(self, filename):

        save_dict = {'policy': self.policy.state_dict(),
                     'target_policy': self.target_policy.state_dict(),
                     'critic': self.critic.state_dict(),
                     'target_critic': self.target_critic.state_dict()}
        torch.save(save_dict, filename)

    def init_comm_from_save(self, filename):

        save_dict = torch.load(filename)
        self.policy.load_state_dict(save_dict['policy'])
        self.target_policy.load_state_dict(save_dict['target_policy'])
        self.critic.load_state_dict(save_dict['critic'])
        self.target_critic.load_state_dict(save_dict['target_critic'])
