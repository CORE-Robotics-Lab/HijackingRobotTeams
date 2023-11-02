import enum
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt

class ReplayBufferPlain(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, fixed_steps, num_agents, obs_dims, ac_dims, comm_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps # 1*(10^6)
        self.fixed_steps = fixed_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.comm_buffs = []
        self.next_comm_buffs = []
        # obs_dims = [26(predator), 26(predator), 26(predator), 24(prey), 24(capturer), 24(capturer)] in pcp environment
        # ac_dims = [5(predator), 5(predator), 5(predator), 5(prey), 6(capturer), 6(capturer)]
        for odim, adim, cdim in zip(obs_dims, ac_dims, comm_dims): # create obs, action, reward, next obs, done buffers for each agent
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))
            self.comm_buffs.append(np.zeros((max_steps, cdim)))
            self.next_comm_buffs.append(np.zeros((max_steps, cdim)))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, comms=None, next_comms=None):
        nentries = observations.shape[0]  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                self.comm_buffs[agent_i] = np.roll(self.comm_buffs[agent_i], rollover, axis=0)
                self.next_comm_buffs[agent_i] = np.roll(self.next_comm_buffs[agent_i], rollover, axis=0)
            self.curr_i = self.fixed_steps
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i] # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
            if comms is not None:
                self.comm_buffs[agent_i][self.curr_i:self.curr_i + nentries] = comms
                self.next_comm_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_comms
        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = self.fixed_steps

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is False by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.comm_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_comm_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

    def get_std_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].std() for i in range(self.num_agents)]
          
class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, agent_types, obs_dims, ac_dims, replay_buffer_record, feature_class_names, is_cuda):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps # 1*(10^6)
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.td_buffs = []
        self.prob_buffs = []
        self.time_buffs = []
        self.feature_min = replay_buffer_record.wrapped_feature_min
        self.feature_max = replay_buffer_record.wrapped_feature_max
        self.interval = replay_buffer_record.wrapped_feature_interval
        self.feature_class_names = feature_class_names
        self.prob_counts = []
        self.probs = []
        self.device = 'cuda' if is_cuda else 'cpu'

        # obs_dims = [26(predator), 26(predator), 26(predator), 24(prey), 24(capturer), 24(capturer)] in pcp environment
        # ac_dims = [5(predator), 5(predator), 5(predator), 5(prey), 6(capturer), 6(capturer)]
        for odim, adim in zip(obs_dims, ac_dims): # create obs, action, reward, next obs, done buffers for each agent
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))
            self.td_buffs.append(np.zeros(max_steps))
            self.prob_buffs.append(np.zeros(max_steps))
            self.time_buffs.append(np.zeros(max_steps))
        
        # INFO: self.prob_counts and self.probs should be with size [num_agents, num_feature]
        for ag_idx in range(self.num_agents):
            # self.feature_bin_num = int(np.ceil((self.feature_max - self.feature_min) / feature_interval))
            # prob_map_dims = (self.num_agents, self.feature_class_num, )    
            # INFO: initialize the probability map counts and probs
            self.prob_counts.append([np.zeros_like(np.arange(self.feature_min[ag_idx][feat_idx], self.feature_max[ag_idx][feat_idx] + 1e-6, self.interval[ag_idx][feat_idx])) for feat_idx in range(len(self.feature_class_names))])
            self.probs.append([np.ones_like(np.arange(self.feature_min[ag_idx][feat_idx], self.feature_max[ag_idx][feat_idx] + 1e-6, self.interval[ag_idx][feat_idx])) * 1e-6 for feat_idx in range(len(self.feature_class_names))])
            self.feature_max_in_buffer = -100.0 * np.ones((self.num_agents, len(self.feature_class_names)))
        # INFO: initialize the occupancy (show if this position WAS occupied before [0: not occupied])
        self.occupancy = np.zeros(max_steps)

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, sharing_td, sharing_prob, t):
        nentries = observations.shape[0]  # handle multiple parallel environments (num of environments)
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents): # np.roll: rolling end to start by the unit of rollover
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
                self.td_buffs[agent_i] = np.roll(self.td_buffs[agent_i], rollover)
                self.prob_buffs[agent_i] = np.roll(self.prob_buffs[agent_i], rollover)
                self.time_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)

            # INFO: roll the occupancy
            self.occupancy = np.roll(self.occupancy, rollover)
                
            self.curr_i = 0
            self.filled_i = self.max_steps
        
        occupancy_update = self.occupancy[self.curr_i:self.curr_i + nentries]
        for idx, occ in enumerate(occupancy_update):
            if occ == 1:
                for agent_i in range(self.num_agents):
                    for feat_idx, feature_class_name in enumerate(self.feature_class_names):
                        if feature_class_name != "td_error":
                            feature = self.feature_select_from_buffer(feature_class_name, agent_i, self.curr_i, nentries)
                            # self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries][idx]
                            interval_idx_of_feat = self.cal_bin_idx(feature, self.feature_min[agent_i][feat_idx], self.feature_max[agent_i][feat_idx], self.interval[agent_i][feat_idx])
                            self.prob_counts[agent_i][feat_idx][interval_idx_of_feat] = self.prob_counts[agent_i][feat_idx][interval_idx_of_feat] - 1
            else:
                continue

        # INFO: update the occupancy
        self.occupancy[self.curr_i:self.curr_i + nentries] = 1

        for agent_i in range(self.num_agents): # for each agent
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i]) # self.obs_buffs[0].shape = (1000000, 26) and self.obs_buffs[][] is a row (26, ). Also observations[,] is an array of array and np.vstack() remove the outer array
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
            self.td_buffs[agent_i][self.curr_i:self.curr_i + nentries] = sharing_td[:, agent_i]
            self.prob_buffs[agent_i][self.curr_i:self.curr_i + nentries] = sharing_prob[:, agent_i]
            self.time_buffs[agent_i][self.curr_i:self.curr_i + nentries] = t[:, agent_i]
            # INFO: add new prob counts to calculate selection prob
            for feat_idx, feature_class_name in enumerate(self.feature_class_names):
                if feature_class_name != "td_error":
                    feature = self.feature_select_from_buffer(feature_class_name, agent_i, self.curr_i, nentries)
                    self.feature_max_in_buffer[agent_i][feat_idx] = np.maximum(self.feature_max_in_buffer[agent_i][feat_idx], feature)
                    interval_idices_of_feat = self.cal_bin_idx(feature, self.feature_min[agent_i][feat_idx], self.feature_max[agent_i][feat_idx], self.interval[agent_i][feat_idx])
                    self.prob_counts[agent_i][feat_idx][interval_idices_of_feat] = self.prob_counts[agent_i][feat_idx][interval_idices_of_feat] + 1
                else:
                    pass

        self.curr_i += nentries # one enviornment here, so plus one in each invoking of push
        if self.filled_i < self.max_steps: # how many positions in the buffer have been filled?
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0



    def feature_select_from_buffer(self, sub_buffer_name, agent_idx, feature_loc_idx, nentries):
        if sub_buffer_name == "reward":
            selected_feature = self.rew_buffs[agent_idx][feature_loc_idx:feature_loc_idx + nentries]
        elif sub_buffer_name == "td_error":
            selected_feature = self.td_buffs[agent_idx][feature_loc_idx:feature_loc_idx + nentries]
        else:
            raise NotImplementedError
        return selected_feature

    def clip_to_range(self, features, min_val, max_val):
        return np.clip(features, min_val, max_val) - min_val

    def cal_bin_idx(self, features, min_val, max_val, interval):
        clipped_feature = np.clip(features, min_val, max_val) - min_val
        bin_idx = (np.floor(clipped_feature / interval)).astype(np.int)
        return bin_idx

    def generate_prob_map(self, video_path, episode, record_agents_indices):

        for agent_i in record_agents_indices: # for each agent need to be recorded
            for feat_idx, feature_class_name in enumerate(self.feature_class_names):
                # INFO: update the probs
                self.probs[agent_i][feat_idx] = self.clip_to_range(self.prob_counts[agent_i][feat_idx] / self.filled_i, 1e-6, 1.0)

        for ag_idx in record_agents_indices: # for each agent need to be recorded
            for feat_idx, feature_class_name in enumerate(self.feature_class_names):
                prob_bins = np.arange(self.feature_min[ag_idx][feat_idx], self.feature_max[ag_idx][feat_idx] + 1e-6, self.interval[ag_idx][feat_idx])
                prob = self.probs[ag_idx][feat_idx]
                plt.bar(prob_bins, prob, width=self.interval[ag_idx][feat_idx])
                plt.title("Episode %i: agent%i_%s, history_max = %f" % (episode, ag_idx, feature_class_name, self.feature_max_in_buffer[ag_idx][feat_idx]))
                plt.savefig(str(video_path) + ("_agent%i_%s" % (ag_idx, feature_class_name)))
                plt.close()

    def get_prob(self, agent_i, feature):
        interval_idx_of_rew = torch.floor((torch.clip(feature, self.feature_min, self.feature_max) - self.feature_min)/self.interval) 
        prob = self.probs[agent_i][interval_idx_of_rew.cpu().detach().numpy().astype(int)]
        return prob

    def sample(self, N, to_gpu=False, norm_rews=True):
        norm_rews = False
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False) # choose N from self.filled_i, N is the batch size, it returns an np array with dimension (N,)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews: # it is False by default
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)] # For each agent, save the normalized rewards definded by inds into a list
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]) # return 5 lists whose position is defined by inds

    def asyn_sample(self, N, to_gpu=False, norm_rews=False, mode="random"): # sample one batch.
        ret_rews = []
        obs = []
        ac = []
        next_obs = []
        done_buff = []
        td_buffs = []
        prob_buffs = []
        t_buffs = []

        # inds_adv = np.random.choice(np.arange(self.filled_i), size=N, replace=False) # shape: (1024,) - one batch
        # inds_prey = np.random.choice(np.arange(self.filled_i), size=N, replace=False) 
        # inds_capture = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        if mode == "random": 
            inds = [np.random.choice(np.arange(self.filled_i), size=np.min([N,self.filled_i]), replace=False)]
        elif mode == "recent":
            if self.filled_i == self.max_steps:
                inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
            else:
                inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        else:
            raise NotImplementedError
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        for i in range(self.num_agents):
            # if self.agent_types[i] == 'adversary':
            #     inds = inds_adv
            # elif self.agent_types[i] == 'prey':
            #     inds = inds_prey
            # else:
            #     inds = inds_capture
            
            if norm_rews: # it is true by default
                ret_rews.append(cast((self.rew_buffs[i][inds] -
                                self.rew_buffs[i][:self.filled_i].mean()) /
                                self.rew_buffs[i][:self.filled_i].std())) # For each agent, save the normalized rewards definded by inds into a list
            else:
                ret_rews.append(cast(self.rew_buffs[i][inds]))

            obs.append(cast(self.obs_buffs[i][inds])) # [adv1_obs_inds_adv (1024 elements(one batch), same for the nexts), adv2_obs_inds_adv, adv3_obs_inds_adv, prey_obs_inds_prey, cap1_obs_inds_capture, cap2_obs_inds_capture]
            ac.append(cast(self.ac_buffs[i][inds]))
            next_obs.append(cast(self.next_obs_buffs[i][inds]))
            done_buff.append(cast(self.done_buffs[i][inds]))
            td_buffs.append(cast(self.td_buffs[i][inds]))
            prob_buffs.append(cast(self.prob_buffs[i][inds]))   
            t_buffs.append(cast(self.time_buffs[i][inds]))

        # stat_agent_id = 5

        # time_td_rew_num = [torch.mean(t_buffs[stat_agent_id]), torch.mean(td_buffs[stat_agent_id]), torch.mean(ret_rews[stat_agent_id])]     

        return inds, obs, ac, ret_rews, next_obs, done_buff, td_buffs, prob_buffs, t_buffs #, time_td_rew_num  # return 5 lists whose position is defined by inds


    def heter_sample(self, N, agent_ind, to_gpu=False, norm_rews=True): # sample one batch.
        ret_rews = []
        obs = []
        ac = []
        next_obs = []
        done_buff = []
        td_buffs = []
        prob_buffs = []
        t_buffs = []


        # inds = [np.random.choice(np.arange(self.filled_i), size=N, replace=False)]
        inds_sharing_sel = np.nonzero(np.random.binomial(size=self.filled_i, n=1, p=self.prob_buffs[agent_ind][0:self.filled_i]))
        if inds_sharing_sel[0].shape[0] == 0 or inds_sharing_sel[0].shape[0] == 1:
            inds_sharing_sel = [np.array([0, 1])]
        inds = [np.random.choice(inds_sharing_sel[0], size=np.min([N,inds_sharing_sel[0].shape[0]]), replace=False)]
        # inds = [np.arange(self.filled_i)]
        # inds = [inds_sharing_sel[0]]

        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        for i in range(self.num_agents):
            # if self.agent_types[i] == 'adversary':
            #     inds = inds_adv
            # elif self.agent_types[i] == 'prey':
            #     inds = inds_prey
            # else:
            #     inds = inds_capture
            
            if norm_rews: # it is true by default
                ret_rews.append(cast((self.rew_buffs[i][inds] -
                                self.rew_buffs[i][:self.filled_i].mean()) /
                                self.rew_buffs[i][:self.filled_i].std())) # For each agent, save the normalized rewards definded by inds into a list
            else:
                ret_rews.append(cast(self.rew_buffs[i][inds]))

            obs.append(cast(self.obs_buffs[i][inds])) # [adv1_obs_inds_adv (1024 elements(one batch), same for the nexts), adv2_obs_inds_adv, adv3_obs_inds_adv, prey_obs_inds_prey, cap1_obs_inds_capture, cap2_obs_inds_capture]
            ac.append(cast(self.ac_buffs[i][inds]))
            next_obs.append(cast(self.next_obs_buffs[i][inds]))
            done_buff.append(cast(self.done_buffs[i][inds]))
            td_buffs.append(cast(self.td_buffs[i][inds]))
            prob_buffs.append(cast(self.prob_buffs[i][inds]))     
            t_buffs.append(cast(self.time_buffs[i][inds]))     

        return inds, obs, ac, ret_rews, next_obs, done_buff, td_buffs, prob_buffs, t_buffs  # return 5 lists whose position is defined by inds

    def seq_sample(self, N, seq_len, to_gpu=False, norm_rews=False): # sample one batch of sequence.
        ret_rews = []
        obs = []
        ac = []
        next_obs = []
        done_buff = []
        td_buffs = []
        prob_buffs = []
        t_buffs = []
        n = 0
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        inds_end = [np.maximum(np.random.choice(np.arange(self.filled_i), size=np.min([N,self.filled_i]), replace=False), seq_len)]
        # inds_start = [inds_end[0] - seq_len]
        for a in range(self.num_agents):
            obs_dim = self.obs_buffs[a].shape[-1]
            act_dim = self.ac_buffs[a].shape[-1]
            batch_seq_rew = cast(torch.zeros(N, seq_len))
            batch_seq_obs = cast(torch.zeros(N, seq_len, obs_dim))
            batch_seq_act = cast(torch.zeros(N, seq_len, act_dim))
            batch_seq_nextObs = cast(torch.zeros(N, seq_len, obs_dim))
            batch_seq_done = cast(torch.zeros(N, seq_len))
            batch_seq_td = cast(torch.zeros(N, seq_len))
            batch_seq_prob = cast(torch.zeros(N, seq_len))
            batch_seq_t = cast(torch.zeros(N, seq_len))
            for i in inds_end[0]:
                for j in range(seq_len):
                    if (self.done_buffs[a][inds_end-1-j]) is True and j != 0:
                        break
                    if norm_rews:
                        batch_seq_rew[n,seq_len-j] = (self.rew_buffs[a][inds_end-1-j] - self.rew_buffs[a][:self.filled_i].mean()) / self.rew_buffs[i][:self.filled_i].std()
                    else:
                        batch_seq_rew[n,seq_len-j] = self.rew_buffs[a][inds_end-1-j]
                    batch_seq_obs[n,seq_len-j,:] = self.obs_buffs[a][inds_end-1-j]
                    batch_seq_act[n,seq_len-j,:] = self.ac_buffs[a][inds_end-1-j]
                    batch_seq_nextObs[n,seq_len-j,:] = self.next_obs_buffs[a][inds_end-1-j]
                    batch_seq_done[n,seq_len-j,:] = self.done_buffs[a][inds_end-1-j]
                    batch_seq_td[n,seq_len-j,:] = self.td_buffs[a][inds_end-1-j]
                    batch_seq_prob[n,seq_len-j,:] = self.prob_buffs[a][inds_end-1-j]
                    batch_seq_t[n,seq_len-j,:] = self.time_buffs[a][inds_end-1-j]
                n = n + 1
            ret_rews.append(batch_seq_rew)
            obs.append(batch_seq_obs)
            ac.append(batch_seq_act)
            next_obs.append(batch_seq_nextObs)
            done_buff.append(batch_seq_done)
            td_buffs.append(batch_seq_td)
            prob_buffs.append(batch_seq_prob)
            t_buffs.append(batch_seq_t)
        return inds_end, obs, ac, ret_rews, next_obs, done_buff, td_buffs, prob_buffs, t_buffs #, time_td_rew_num  # return 5 lists whose position is defined by inds

    def update_td_error(self, agent_ind, buffer_inds, td_error):
        td_prev = self.td_buffs[agent_ind][buffer_inds]
        td_prev_nonzero = td_prev[np.nonzero(td_prev)]

        self.td_buffs[agent_ind][buffer_inds] = td_error.squeeze(1).data.cpu().numpy()
        # INFO: Update the td_error counts for the probability map
        feature_class_name = "td_error"
        feature = self.td_buffs[agent_ind][buffer_inds]
        feat_idx = self.feature_class_names.index(feature_class_name)
        self.feature_max_in_buffer[agent_ind][feat_idx] = np.max(np.maximum(self.feature_max_in_buffer[agent_ind][feat_idx], feature))
        
        if len(td_prev_nonzero) != 0: 
            interval_idices_of_feat_prev = self.cal_bin_idx(td_prev_nonzero, self.feature_min[agent_ind][feat_idx], self.feature_max[agent_ind][feat_idx], self.interval[agent_ind][feat_idx])
            interval_idx_prev, counts_prev = np.unique(interval_idices_of_feat_prev, return_counts=True)
            self.prob_counts[agent_ind][feat_idx][interval_idx_prev] = self.prob_counts[agent_ind][feat_idx][interval_idx_prev] - counts_prev

        interval_idices_of_feat = self.cal_bin_idx(feature, self.feature_min[agent_ind][feat_idx], self.feature_max[agent_ind][feat_idx], self.interval[agent_ind][feat_idx])
        interval_idx, counts = np.unique(interval_idices_of_feat, return_counts=True)
        self.prob_counts[agent_ind][feat_idx][interval_idx] = self.prob_counts[agent_ind][feat_idx][interval_idx] + counts
        return



    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

    def get_std_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].std() for i in range(self.num_agents)]


class ReplayBufferRecord(object):
    def __init__(self, agent_num, feature_num, configs_agents_feats_minMaxIntervals) -> None:
        self.agent_num = agent_num
        self.feature_num = feature_num
        self.configs_agents_feats_minMaxIntervals = configs_agents_feats_minMaxIntervals
        self.feature_min = np.zeros((self.agent_num, self.feature_num))
        self.feature_max = np.ones((self.agent_num, self.feature_num))
        self.feature_interval = np.ones((self.agent_num, self.feature_num)) * 0.1
    
    @property
    def wrapped_feature_min(self):
        for config_idx, agents_feats_minmaxinterval in enumerate(self.configs_agents_feats_minMaxIntervals):
            for agent_idx in agents_feats_minmaxinterval[0]:
                for i, feat_idx in enumerate(agents_feats_minmaxinterval[1]):
                    self.feature_min[agent_idx][feat_idx] = agents_feats_minmaxinterval[2][i][0]
        return self.feature_min

    @property
    def wrapped_feature_max(self):
        for config_idx, agents_feats_minmaxinterval in enumerate(self.configs_agents_feats_minMaxIntervals):
            for agent_idx in agents_feats_minmaxinterval[0]:
                for i, feat_idx in enumerate(agents_feats_minmaxinterval[1]):
                    self.feature_max[agent_idx][feat_idx] = agents_feats_minmaxinterval[2][i][1]
        return self.feature_max

    @property
    def wrapped_feature_interval(self):
        for config_idx, agents_feats_minmaxinterval in enumerate(self.configs_agents_feats_minMaxIntervals):
            for agent_idx in agents_feats_minmaxinterval[0]:
                for i, feat_idx in enumerate(agents_feats_minmaxinterval[1]):
                    self.feature_interval[agent_idx][feat_idx] = agents_feats_minmaxinterval[2][i][2]
        return self.feature_interval