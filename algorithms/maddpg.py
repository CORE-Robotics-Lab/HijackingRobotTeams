import enum
import copy
import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, two_hot_encode, direct_encode, soft_encode, get_agents_SA_pairs
from utils.agents import DDPGAgent
import functorch

MSELoss = torch.nn.MSELoss()
MSELoss_each = torch.nn.MSELoss(reduction='none')

class MADDPG(object):
    def __init__(self, agent_init_params, alg_types, agent_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, is_cuda = True):

        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agent_types = agent_types
        self.device = 'cuda' if is_cuda else 'cpu'
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, device=self.device,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]
    
    @property
    def surrogate_policies(self):
        return [a.surrogate_policy for a in self.agents]

    @property
    def target_surrogate_policies(self):
        return [a.target_surrogate_policy for a in self.agents]

    @property
    def discriminators(self):
        return [a.discriminator for a in self.agents]

    @property
    def reward_observers(self):
        return [a.reward_observer for a in self.agents]
        
    def scale_noise(self, scale):

        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):

        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, rl_ratio, mode="homo", parallel=False, logger=None, is_debug=True):

        if mode == "homo":
            inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, t_buffs = sample
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = sample[0]
            idxs = sample[1]
            rl_ratio = sample[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = sample # len(next_obs)=agent total num



        curr_agent = self.agents[agent_i]
        curr_type = self.agent_types[agent_i]

        obs_type = []
        next_obs_type = []
        acs_type = []

        for ind, obsa in enumerate(obs):
            if self.agent_types[ind] == curr_type:
                obs_type.append(obsa)

        for ind, obsa in enumerate(next_obs):
            if self.agent_types[ind] == curr_type:
                next_obs_type.append(obsa)

        for ind, acsa in enumerate(acs):
            if self.agent_types[ind] == curr_type:
                acs_type.append(acsa)                

        if is_debug and curr_type == "capturer":
            # print("obs_type[0][0] = ", obs_type[0][0])
            # print("obs_type_shape[0][0] = ", len(obs_type[0][0]))
            pass
        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            trgt_acs_type = []
            if self.discrete_action: # one-hot encode action
                
                all_trgt_acs = [two_hot_encode(explore=False, num_in_comm=ag.num_in_comm, num_in_move=ag.num_in_move, action=pi(nobs)) for pi, nobs, ag in
                                zip(self.target_policies, next_obs, self.agents)] # (pi, next_obs, target_critic)->reward_prediction and next_obs[0].shape=[batch_size, obs_dim(22/20)]
                for ind, trgta in enumerate(all_trgt_acs):
                    if self.agent_types[ind] == curr_type:
                        trgt_acs_type.append(trgta)                                
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
                
 
            trgt_vf_in = torch.cat((*next_obs_type, *trgt_acs_type), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs_type, *acs_type), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss_each = MSELoss_each(actual_value, target_value.detach())
        td_error_each = target_value - actual_value
        td_error_abs_each = torch.abs(td_error_each)

        # vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss = torch.mean(vf_loss_each.squeeze() * torch.Tensor(rl_ratio).to(device=self.device))
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        # curr_agent.scheduler_critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = two_hot_encode(explore=True, num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_pol_out)
            # curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            all_pol_acs_type = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    ag = self.agents[i]
                    all_pol_acs.append(two_hot_encode(explore=False, num_in_comm=ag.num_in_comm, num_in_move=ag.num_in_move, action=pi(ob)))
                    
                    # all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))

            for ind, trgta in enumerate(all_pol_acs):
                if self.agent_types[ind] == curr_type:
                    all_pol_acs_type.append(trgta)                     
            vf_in = torch.cat((*obs_type, *all_pol_acs_type), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        # pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        # curr_agent.scheduler_policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': torch.mean(vf_loss_each),
                                'td_error': torch.mean(td_error_each),
                                'pol_loss': pol_loss},
                               self.niter)

        return td_error_abs_each

    def update_surrogate_policies(self, recent_samples, revised_obs_list, agent_i, update_num, step_len=0.2, mode="homo", parallel=False, logger=None, is_debug=True):
        if mode == "homo":
            inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, t_buffs = recent_samples
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = recent_samples[0]
            idxs = recent_samples[1]
            rl_ratio = recent_samples[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = recent_samples # len(next_obs)=agent total num

        curr_agent = self.agents[agent_i]
        curr_type = self.agent_types[agent_i]

        obs_type = []
        next_obs_type = []
        acs_type = []

        for ind, obsa in enumerate(obs):
            if self.agent_types[ind] == curr_type:
                obs_type.append(obsa)

        for ind, obsa in enumerate(next_obs):
            if self.agent_types[ind] == curr_type:
                next_obs_type.append(obsa)

        for ind, acsa in enumerate(acs):
            if self.agent_types[ind] == curr_type:
                acs_type.append(acsa)                

        if is_debug and curr_type == "capturer":
            # print("obs_type[0][0] = ", obs_type[0][0])
            # print("obs_type_shape[0][0] = ", len(obs_type[0][0]))
            pass

        obs = revised_obs_list.copy()

        def add_noise_to_obs(unnoised_obs):
            unnoised_obs = copy.deepcopy(unnoised_obs)
            obs_noise = step_len * (torch.rand_like(unnoised_obs[:,:curr_agent.obs_num_in_obs]) - 0.5)
            comm_noise_indices = (torch.rand_like(unnoised_obs[:,curr_agent.obs_num_in_obs:]) < step_len)
            noised_obs = unnoised_obs[:,:curr_agent.obs_num_in_obs] + obs_noise
            unnoised_obs[:,curr_agent.obs_num_in_obs:][comm_noise_indices] = 1 - unnoised_obs[:,curr_agent.obs_num_in_obs:][comm_noise_indices]
            noised_comm = unnoised_obs[:,curr_agent.obs_num_in_obs:]
            return torch.cat((noised_obs, noised_comm), dim=-1)

        soft_surrogate_action_list = []
        surrogate_action_list = []
        for update_i in range(update_num):
            if update_i == 0:
                obs_in = obs[agent_i].detach()
            else:
                obs_in = add_noise_to_obs(obs[agent_i]).detach()
            curr_agent.surrogate_policy_optimizer.zero_grad()
            soft_surrogate_action = direct_encode(action=curr_agent.surrogate_policy(obs_in))
            soft_gt_action = direct_encode(action=curr_agent.policy(obs_in))
            soft_surrogate_loss = MSELoss(soft_surrogate_action, soft_gt_action.detach())
            soft_surrogate_action_list.append(soft_surrogate_loss.detach())

            surrogate_action = two_hot_encode(explore=False, num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.surrogate_policy(obs_in))
            gt_action = two_hot_encode(explore=False, num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.policy(obs_in))
            surrogate_loss = MSELoss(surrogate_action, gt_action.detach())  
            surrogate_action_list.append(surrogate_loss.detach())  
            if update_i == 0:
                unnoised_surrogate_loss = surrogate_loss.detach()

            soft_surrogate_loss.backward()
            if parallel:
                average_gradients(curr_agent.surrogate_policy)
            torch.nn.utils.clip_grad_norm(curr_agent.surrogate_policy.parameters(), 0.5)
            curr_agent.surrogate_policy_optimizer.step()
        surrogate_loss_mean = torch.vstack(surrogate_action_list).mean()
        soft_surrogate_loss_mean = torch.vstack(soft_surrogate_action_list).mean()

        if logger is not None:
            logger.add_scalars('agent%i/surrogate_losses' % agent_i, {'surrogate_train_loss': surrogate_loss_mean, 'unnoised_surrogate_train_loss': unnoised_surrogate_loss, 'soft_surrogate_train_loss': soft_surrogate_loss_mean}, self.niter)
        return 

    def update_surrogate_policies_GAN(self, recent_samples, agent_i, mode="homo", parallel=False, logger=None, is_debug=True):
        if mode == "homo":
            inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, t_buffs = recent_samples
        elif mode == "per":
            obs, acs, rews, next_obs, dones, td_error = recent_samples[0]
            idxs = recent_samples[1]
            rl_ratio = recent_samples[2]
        else:
            inds, obs, acs, rews, next_obs, dones, inds_sharing_sel = recent_samples # len(next_obs)=agent total num

        curr_agent = self.agents[agent_i]
        curr_type = self.agent_types[agent_i]        
        
        surrogate_action = two_hot_encode(explore=True, num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.surrogate_policy(obs[agent_i]))
        
        real_sample = torch.cat((obs[agent_i], acs[agent_i]), dim=-1)
        fake_sample = torch.cat((obs[agent_i], surrogate_action), dim=-1)

        # Update the discriminator
        curr_agent.discriminator_optimizer.zero_grad()
        discriminitor_loss = (-torch.log(curr_agent.discriminator(real_sample)[:,0])).mean() + (-torch.log(1 - curr_agent.discriminator(fake_sample.detach())[:,0])).mean()
        discriminitor_loss.backward()
        curr_agent.discriminator_optimizer.step()
        # Update the generator
        curr_agent.surrogate_policy_optimizer.zero_grad()
        generator_loss = torch.log(1 - curr_agent.discriminator(fake_sample)[:,0]).mean()
        generator_loss.backward()
        curr_agent.surrogate_policy_optimizer.step()
        # Test
        surrogate_loss = MSELoss(surrogate_action, acs[agent_i])

        if parallel:
            average_gradients(curr_agent.discriminator)
            average_gradients(curr_agent.surrogate_policy)
        torch.nn.utils.clip_grad_norm(curr_agent.discriminator.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm(curr_agent.surrogate_policy.parameters(), 0.1)

        if logger is not None:
            logger.add_scalars('agent%i/GAN_losses' % agent_i, {'discriminator_loss': discriminitor_loss, 'generator_loss': generator_loss, 'surrogate_train_loss': surrogate_loss}, self.niter)
        return 

    def update_reward_observers_GAN(self, recent_samples, agent_i, logger=None):
        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, t_buffs = recent_samples # len(next_obs)=agent total num
        curr_agent = self.agents[agent_i]

        real_SA_paris = get_agents_SA_pairs(obs, acs)
        fake_agent_actions = two_hot_encode(explore=True, num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.surrogate_policy(obs[agent_i]))
        fake_SA_pair = torch.cat((obs[agent_i], fake_agent_actions), dim=-1)

        real_rew = curr_agent.reward_observer(real_SA_paris[agent_i])
        fake_rew = curr_agent.reward_observer(fake_SA_pair)

        real_discriminator_res = torch.exp(real_rew) / (torch.exp(real_rew) + soft_encode(num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.surrogate_policy(obs[agent_i]))[acs[agent_i]==1].unsqueeze(1).detach()) 
        fake_discriminator_res = torch.exp(fake_rew) / (torch.exp(fake_rew) + soft_encode(num_in_comm=curr_agent.num_in_comm, num_in_move=curr_agent.num_in_move, action=curr_agent.surrogate_policy(obs[agent_i]))[fake_agent_actions==1].unsqueeze(1).detach()) 
        # Update the discriminator
        curr_agent.reward_observer.zero_grad()
        discriminitor_loss = (-torch.log(real_discriminator_res)).mean() + (-torch.log(1 - fake_discriminator_res)).mean()
        discriminitor_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.reward_observer.parameters(), 0.25)
        curr_agent.reward_optimizer.step()
        curr_agent.scheduler_reward_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/GAN_losses' % agent_i, {'discriminator_loss': discriminitor_loss}, self.niter)
        return

    def update_gh(self, recent_samples, agent_i, logger=None):
        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, t_buffs = recent_samples # len(next_obs)=agent total num
        curr_agent = self.agents[agent_i]
        f = (torch.log(soft_encode(num_in_comm=self.agents[agent_i].num_in_comm, num_in_move=self.agents[agent_i].num_in_move, action=self.agents[agent_i].policy(obs[agent_i]))[acs[agent_i]==1]))
        f_approx = curr_agent.airl_discriminator.f(obs[agent_i], self.agents[agent_i].policy(obs[agent_i])[acs[agent_i]==1], dones[agent_i], next_obs[agent_i])
        f_loss = MSELoss(f_approx, f)
        curr_agent.airl_optimizer.zero_grad()
        f_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.airl_discriminator.parameters(), 0.5)
        curr_agent.airl_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/GAN_losses' % agent_i, {'discriminator_loss': f_loss}, self.niter)
        return

    def get_GAN_reward(self, obs, acs, agents_SA_pairs, maddpg, recv_agents_indices):
        discriminator_rewards = []
        for agent_i in recv_agents_indices:
            agent_SA_pair = torch.cat((obs[agent_i], acs[agent_i]), dim=-1)
            rew = maddpg.reward_observers[agent_i](agent_SA_pair)
            discriminator_res = torch.exp(rew) / (torch.exp(rew) + soft_encode(num_in_comm=maddpg.agents[agent_i].num_in_comm, num_in_move=maddpg.agents[agent_i].num_in_move, action=maddpg.agents[agent_i].surrogate_policy(obs[agent_i]))[acs[agent_i]==1].unsqueeze(1).detach()) 
            discriminator_reward = (torch.log(discriminator_res)).mean() + (-torch.log(1 - discriminator_res)).mean()
            discriminator_rewards.append(discriminator_reward)
        return discriminator_reward

    def get_td_error(self, sample, agent_i, is_cuda):

        if is_cuda:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)

        obs, acs, rews, next_obs, dones = sample # len(next_obs)=agent total num
        obs = [cast(obs[i]).unsqueeze(0) for i in range(len(obs))]
        acs = [cast(acs[i]).squeeze().unsqueeze(0) for i in range(len(acs))]
        rews = cast(rews)
        next_obs = [cast(next_obs[i]).unsqueeze(0) for i in range(len(next_obs))]
        dones = cast(dones)


        # np.vstack(obs[:, agent_i])
        curr_agent = self.agents[agent_i]
        curr_type = self.agent_types[agent_i]

        obs_type = []
        next_obs_type = []
        acs_type = []

        for ind, obsa in enumerate(obs):
            if self.agent_types[ind] == curr_type:
                obs_type.append(obsa)

        for ind, obsa in enumerate(next_obs):
            if self.agent_types[ind] == curr_type:
                next_obs_type.append(obsa)

        for ind, acsa in enumerate(acs):
            if self.agent_types[ind] == curr_type:
                acs_type.append(acsa)                

        if self.alg_types[agent_i] == 'MADDPG':
            trgt_acs_type = []
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)] # (pi, next_obs, target_critic)->reward_prediction and next_obs[0].shape=[batch_size, obs_dim(22/20)]
                for ind, trgta in enumerate(all_trgt_acs):
                    if self.agent_types[ind]==curr_type:
                        trgt_acs_type.append(trgta)                                
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
                
 
            trgt_vf_in = torch.cat((*next_obs_type, *trgt_acs_type), dim=1) # (pi, next_obs, target_critic)->reward_prediction
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1))) # current_reward_in_buffer + reward_prediction(from t+1)

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs_type, *acs_type), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in) # reward_prediction(from t)

        vf_loss = MSELoss(actual_value, target_value.detach())

        return vf_loss



    def gen_mask(self, agent_ind, sample, time_step):
        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, time_buffs = sample
        filled_len = inds[0].shape[0]
        curr_agent = self.agents[agent_ind]

        select_sample_features = torch.cat(((time_step * torch.ones(filled_len)).unsqueeze(1).cuda(), time_buffs[agent_ind].unsqueeze(1), rews[agent_ind].unsqueeze(1), td_buffs[agent_ind].unsqueeze(1)), 1)
        prob_buffs_test = curr_agent.sharing_policy(select_sample_features.detach()).squeeze(1)
        
        mask_np = np.random.binomial(size=filled_len, n=1, p=prob_buffs_test.data.cpu().numpy()) == 1
        if np.sum(mask_np) == 0 or np.sum(mask_np) == 1:
            mask_np[0] = True
            mask_np[1] = True
        mask_torch = torch.Tensor(mask_np) == 1  

        # inds[0] = inds[0][mask_np]

        time_td_rew_num = [torch.mean(time_buffs[agent_ind][mask_torch]), torch.mean(td_buffs[agent_ind][mask_torch]), torch.mean(rews[agent_ind][mask_torch]), np.sum(mask_np)]
        print("prob_buffs_test: min, mean, max = (%f, %f, %f)" % (prob_buffs_test.min(), prob_buffs_test.mean(), prob_buffs_test.max()))
        return mask_np, prob_buffs_test, time_td_rew_num

    def sampleStat(self, agent_ind, sample):
        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, time_buffs = sample
        time_td_rew = [torch.mean(time_buffs[agent_ind]), torch.mean(td_buffs[agent_ind]), torch.mean(rews[agent_ind])]
        return time_td_rew

    def select_sample(self, agent_ind, sample, time_step, mask_np_all, batch_size):
        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, time_buffs = copy.deepcopy(sample)
        # filled_len=inds[0].shape[0]
        # curr_agent = self.agents[agent_ind]

        # select_sample_features = torch.cat(((time_step * torch.ones(filled_len)).unsqueeze(1), time_buffs[agent_ind].unsqueeze(1), rews[agent_ind].unsqueeze(1), td_buffs[agent_ind].unsqueeze(1)), 1)
        # prob_buffs_test = curr_agent.sharing_policy(select_sample_features.detach()).squeeze(1)
        
        
        # mask_np_all= np.random.binomial(size=filled_len, n=1, p=prob_buffs_test.data.cpu().numpy()) == 1
        inds_sharing_sel = np.nonzero(mask_np_all)
        inds_batch = np.random.choice(inds_sharing_sel[0], size=np.min([inds_sharing_sel[0].shape[0], batch_size]), replace=False)
        temp = np.zeros(mask_np_all.shape)
        temp[inds_batch] = 1
        mask_np = np.array(temp, dtype=bool)

        # if np.sum(mask_np) == 0 or np.sum(mask_np) == 1:
        #     mask_np[0] = True
        #     mask_np[1] = True
        mask_torch = torch.Tensor(mask_np) == 1       

        inds[0] = inds[0][mask_np]
        for i in range(0, self.nagents):
            obs[i] = obs[i][mask_torch]
            acs[i] = acs[i][mask_torch]
            rews[i] = rews[i][mask_torch]
            next_obs[i] = next_obs[i][mask_torch]
            dones[i] = dones[i][mask_torch]
            td_buffs[i] = td_buffs[i][mask_torch]
            time_buffs[i] = time_buffs[i][mask_torch]

        # time_td_rew_num = [torch.mean(time_buffs[i]), torch.mean(td_buffs[i]), torch.mean(rews[i]), np.sum(mask_np)]
        return inds, obs, acs, rews, next_obs, dones, inds_sharing_sel #, prob_buffs_test, mask_np, time_td_rew_num

    def combine_sample(self, samples1, samples2):
        combined_sample = [[] for _ in range(len(samples1))]
        for f_ind, feature in enumerate(samples1):
            for r_ind, feature_rob in enumerate(feature):
                if feature_rob.ndim == 1:
                    combined_sample[f_ind].append(torch.Tensor(np.append(samples1[f_ind][r_ind], samples2[f_ind][r_ind])))
                else:
                    combined_sample[f_ind].append(torch.vstack((samples1[f_ind][r_ind], samples2[f_ind][r_ind])))

        return combined_sample


    def update_sharing_policy(self, agent_i, sharing_sample, cr_pre, cr_cur, mask_np, sharing_probs):

        inds, obs, acs, rews, next_obs, dones, td_buffs, prob_buffs, time_buffs = sharing_sample 
        sharing_R = cr_cur - cr_pre
        curr_agent = self.agents[agent_i]
        # curr_type = self.agent_types[agent_i]
        
        # sharing_buffer_features = torch.cat(((time_step * torch.ones(rews[agent_i].shape[0])).unsqueeze(1), rews[agent_i].unsqueeze(1), td_buffs[agent_i].unsqueeze(1)), 1)

        curr_agent.sharing_policy_optimizer.zero_grad()
        # sharing_probs = curr_agent.sharing_policy(sharing_buffer_features.detach()).squeeze(1)
        cross_entropy_loss = torch.nn.BCELoss()
        cross_entropy_loss_output = sharing_R * cross_entropy_loss(sharing_probs, torch.Tensor(mask_np+0).cuda())
        cross_entropy_loss_output.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.sharing_policy.parameters(), 0.5)
        curr_agent.sharing_policy_optimizer.step()

        return cross_entropy_loss_output



    def update_all_targets(self):

        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        # self.niter += 1

    def update_surrogate_target(self):
        for a in self.agents:
            soft_update(a.target_surrogate_policy, a.surrogate_policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.policy.eval()
            a.critic.eval()
            a.target_policy.eval()
            a.target_critic.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
                a.target_policy = fn(a.target_policy)
                a.critic = fn(a.critic)
                a.target_critic = fn(a.target_critic)
            self.pol_dev = device
            self.trgt_pol_dev = device
            self.critic_dev = device
            self.trgt_critic_dev = device

    def save(self, filename):

        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def init_agent_from_save(self, agent_ind, filename):

        save_dict = torch.load(filename)
        # instance = cls(**save_dict['init_dict'])
        # instance.init_dict = save_dict['init_dict']
        self.agents[agent_ind].load_params(save_dict['agent_params'][agent_ind])
        # for a, params in zip(instance.agents, save_dict['agent_params']):
        #     a.load_params(params)
        # return instance

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, is_cuda="true"):

        agent_init_params = []
        alg_types = []
        agent_types = []
        agent_type_dict = {'adversary': adversary_alg, 'capturer': adversary_alg, 'prey': agent_alg}
        for atype in env.agent_types:
            alg_types.append(agent_type_dict[atype])
            agent_types.append(atype)
            
        for acsp, obsp, algtype, agent_type in zip(env.action_space, env.observation_space, alg_types, agent_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)

            # INFO: Split the communication dim in action
            if isinstance(acsp, MultiDiscrete):
                # num_in_comm = acsp.nvec[1]
                num_in_comm = acsp.high[1] + 1
            else:
                num_in_comm = 0

            if algtype == "MADDPG":
                num_in_critic = 0
                for ind, oobsp in enumerate(env.observation_space): # need to investigate whose observation spaces we should include
                    if agent_types[ind] == agent_type:
                        num_in_critic += oobsp.shape[0] # add observation dimensions agent by agent FOR THE SAME TYPE (22+22+22/20/22+22)
                for ind, oacsp in enumerate(env.action_space): # need to investigate whose action spaces we should include
                    if agent_types[ind] == agent_type:
                        num_in_critic += get_shape(oacsp) # add action dimensions agent by agent FOR THE SAME TYPE (22+22+22/20/22+22)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_comm': num_in_comm})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_types': agent_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'is_cuda': is_cuda}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):

        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

