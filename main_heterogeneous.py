
import torch
import os
import numpy as np
import copy
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer, ReplayBufferRecord, ReplayBufferPlain
from utils.env_wrappers import DummyVecEnv
from utils.config_loader import config_loader
from algorithms.maddpg import MADDPG
import yaml
import imageio
from utils.agents import AdvCommDDPGAgent
from utils.misc import flip_comm, update_obs_with_comm_vec, obs_to_comm_obs, obs_to_comms

USE_CUDA = torch.cuda.is_available()
device_train = "cuda" if USE_CUDA else "cpu"

def make_parallel_env(env_id, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env._seed(seed + rank * 1000)
            # np.random.seed(seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)]) # function handle of init_env

def run_adv_comm_offpolicy(config):

    # INFO: Prepare working space
    model_dir = Path('./models') / config["environment"]["env_id"] / config["environment"]["model_name"]
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    with open(run_dir / "parameters.yaml", 'w') as para_yaml:
        yaml.dump(config, para_yaml, default_flow_style=False)
    logger = SummaryWriter(str(log_dir))

    # INFO: Prepare environment
    torch.manual_seed(config["environment"]["seed"])
    torch.cuda.manual_seed(config["environment"]["seed"]) 
    # np.random.seed(config["environment"]["seed"])
    if not USE_CUDA:
        torch.set_num_threads(config["train"]["n_training_threads"])
    env = make_parallel_env(config["environment"]["env_id"], 
                            config["environment"]["seed"],
                            config["train"]["discrete_action"]) # env: DummyVecEnv
    maddpg = MADDPG.init_from_env(env, agent_alg=config["train"]["agent_alg"],
                                  adversary_alg=config["train"]["adversary_alg"],
                                  gamma=config["train"]["gamma"],
                                  tau=config["train"]["tau"],
                                  lr=config["train"]["lr"],
                                  hidden_dim=config["train"]["hidden_dim"],
                                  is_cuda=USE_CUDA)

    comm_agent_indices=[0, 1, 2]
    recv_agent_indices = [3, 4]
    comm_ddpg_num_in_pol, comm_ddpg_num_out_pol, comm_ddpg_num_in_critic, comm_combined_num = 0, 0, 0, 0
    for i in range(len(comm_agent_indices+recv_agent_indices)):
        if i in comm_agent_indices:
            comm_ddpg_num_in_pol = comm_ddpg_num_in_pol + maddpg.init_dict["agent_init_params"][i]["num_in_pol"] # + maddpg.init_dict["agent_init_params"][i]["num_out_pol"]
            comm_ddpg_num_out_pol = comm_ddpg_num_out_pol + maddpg.init_dict["agent_init_params"][i]["num_in_comm"]
            comm_ddpg_num_in_critic = comm_ddpg_num_in_critic + maddpg.init_dict["agent_init_params"][i]["num_in_pol"] + maddpg.init_dict["agent_init_params"][i]["num_out_pol"] - maddpg.init_dict["agent_init_params"][i]["num_in_comm"] # + maddpg.init_dict["agent_init_params"][i]["num_in_comm"]
            comm_combined_num = comm_combined_num + maddpg.init_dict["agent_init_params"][i]["num_in_comm"]
        if i in recv_agent_indices:
            comm_ddpg_num_in_pol = comm_ddpg_num_in_pol + (maddpg.init_dict["agent_init_params"][i]["num_in_pol"]) # + maddpg.init_dict["agent_init_params"][i]["num_out_pol"]
            comm_ddpg_num_out_pol = comm_ddpg_num_out_pol + maddpg.init_dict["agent_init_params"][i]["num_in_comm"]
            comm_ddpg_num_in_critic = comm_ddpg_num_in_critic + (22) + maddpg.init_dict["agent_init_params"][i]["num_out_pol"] # + maddpg.init_dict["agent_init_params"][i]["num_in_comm"]

    comm_ddpg = AdvCommDDPGAgent(num_in_pol=comm_ddpg_num_in_pol, num_out_pol=comm_ddpg_num_out_pol, num_in_critic=comm_ddpg_num_in_critic, num_in_comm=comm_ddpg_num_out_pol, comm_agent_indices=[0, 1, 2], gamma=config["train"]["gamma"], tau=config["train"]["tau"], device="cuda", hidden_dim=256, lr=config["train"]["lr"], discrete_action=True)
    # if config["train"]["continue"]:
    #     comm_ddpg.init_comm_from_save(filename=config["train"]["comm_checkpoint"])    
    if config["environment"]["adversary_env"] == "adversrial":
        if config["environment"]["debug_mode"]:
            print("Execute the adversary environment")
        adversary_ind = [0,1,2,3,4,5]
        prey_ind= [3]
        configs_agents_feats_minMaxIntervals = [[[0, 1, 2], [0, 1], [[-1, 0.5, 0.1], [0, 5, 0.25]]], [[4, 5], [0, 1], [[-1, 1, 0.1], [0, 5, 0.25]]]]
    elif config["environment"]["adversary_env"] == "random_prey":
        if config["environment"]["debug_mode"]:
            print("Execute the random prey policy")
        adversary_ind = [0,1,2,3,4]
        prey_ind = None
        configs_agents_feats_minMaxIntervals = [[[0, 1, 2], [0, 1], [[-1, 0.5, 0.1], [0, 5, 0.25]]], [[3, 4], [0, 1], [[-1, 1, 0.1], [0, 5, 0.25]]]]
    else:
        if config["environment"]["debug_mode"]:
            print("Use a fixed prey policy")
        adversary_ind = [0,1,2,4,5]
        prey_ind= [3]
        maddpg.init_agent_from_save(prey_ind[0], str(model_dir / 'prey' / 'model.pt'))

    # INFO: Initialize replay buffer
    feature_class_names = ["reward", "td_error"]
    replay_buffer_record = ReplayBufferRecord(maddpg.nagents, feature_num=2, configs_agents_feats_minMaxIntervals=configs_agents_feats_minMaxIntervals)
    replay_buffer = ReplayBuffer(config["train"]["buffer_length"], maddpg.nagents, maddpg.agent_types,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space], replay_buffer_record, feature_class_names, USE_CUDA)  
    replay_buffer_comm = ReplayBufferPlain(max_steps=config["train"]["buffer_length"],
                                           fixed_steps=config["train"]["fixed_length"],
                                           num_agents=1,
                                           obs_dims=[comm_ddpg_num_in_pol],
                                           ac_dims=[np.sum([acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space])],
                                           comm_dims=[comm_combined_num])

    # INFO: Prepare to train
    t = 0
    scaled_t = 0
    for ep_i in range(0, config["train"]["n_episodes"], config["train"]["n_rollout_threads"]):
        print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + config["train"]["n_rollout_threads"], config["train"]["n_episodes"]))
        if config["train"]["continue"] and ep_i == config["train"]["continue_start_episode"]:
            maddpg = MADDPG.init_from_save(filename=config["train"]["ag_checkpoint"])
        obs = env.reset(seed = None)
        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False).to(device_train) for i in range(maddpg.nagents)]
        if USE_CUDA:
            maddpg.prep_rollouts(device='gpu')
            comm_ddpg.prep_rollouts(device='cuda')
        else:
            maddpg.prep_rollouts(device='cpu')
            comm_ddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config["train"]["n_exploration_eps"] - ep_i) / config["train"]["n_exploration_eps"]
        maddpg.scale_noise(config["train"]["final_noise_scale"] + (config["train"]["init_noise_scale"] - config["train"]["final_noise_scale"]) * explr_pct_remaining)
        maddpg.reset_noise()

        imgs = []
        for _ in range(config["train"]["episode_length"]):

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False).to(device_train)
                         for i in range(maddpg.nagents)] # torch_obs: [torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16]),torch.Size([1, 16])]

            if ep_i > config["train"]["advcomm_n_episodes"]:
                # INFO: Visualize several episodes after adversarial policy training
                torch_obs_comm = obs_to_comm_obs(torch_obs, obs_len_before_comm=22, blind_agent_indices=[3, 4], mode="normal")
                torch_agent_actions_adv_comm_flip_vec = comm_ddpg.step(torch_obs_comm, explore=False)
                torch_obs, flipped_num = update_obs_with_comm_vec(torch_obs, blind_agent_indices=[3, 4], flip_vec=torch_agent_actions_adv_comm_flip_vec, mode=config["train"]["flip_mode"])
                torch_agent_actions = maddpg.step(torch_obs, explore=False)
            else:
                # INFO: Adversarial policy do NOT change the torch_agent_actions in the training
                torch_agent_actions = maddpg.step(torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config["train"]["n_rollout_threads"])] 
            if ep_i % config["train"]["save_interval"] == 0 and config["train"]["save_gif"]:
                game_img = env.render(render_option='rgb_array')[0]
                imgs.append(game_img)
            next_obs, rewards, dones, infos = env.step(actions)
  
            reward_comm = (-0.5*np.sum(rewards[0,-2:])) 
            td = np.zeros((1,6))
            prob = 0.5*np.ones((1,6))
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones, td, prob, scaled_t*np.ones((1,6)))
            replay_buffer_comm.push(np.expand_dims(np.concatenate(obs[0], axis=-1), axis=0), np.concatenate(agent_actions, axis=-1), np.array([[reward_comm]]), np.expand_dims(np.concatenate(next_obs[0], axis=-1), axis=0), dones)
            obs = next_obs
            # torch_obs_comm = next_torch_obs_comm
            # torch_agent_actions_adv_comm_flip_vec = next_torch_agent_actions_adv_comm_flip_vec
            # reward_comm = next_reward_comm
            t += config["train"]["n_rollout_threads"] # t is initialize outside the n_episode loop
            scaled_t = t/(config["train"]["episode_length"] * config["train"]["n_episodes"])

        if ep_i % config["train"]["save_interval"] == 0 and config["train"]["save_gif"]:
            imageio.mimsave(str(log_dir / ('episode_%i.gif' % (ep_i))), imgs, duration=1/config["train"]["fps"])
        if len(replay_buffer) >= config["train"]["batch_size"]: # update every config["train"]["steps_per_update"] steps
            for u_i in range(config["train"]["n_rollout_threads"]):
                rl_ratio = np.ones(config["train"]["batch_size"])
                samples_ag_timeTdRew = []

                if ep_i < config["train"]["advcomm_n_episodes"]:
                    if USE_CUDA:
                        maddpg.prep_rollouts(device='gpu')
                    else:
                        maddpg.prep_rollouts(device='cpu')
                    sample_comm = replay_buffer_comm.sample(config["train"]["batch_size"], to_gpu=USE_CUDA)
                    agent_obs_with_comm_list = comm_ddpg.update_with_offpolicy(sample_comm, a_i, maddpg, rl_ratio, flip_mode=config["train"]["flip_mode"], mode="homo", logger=logger, is_debug=config["environment"]["debug_mode"])
                    comm_ddpg.update_comm_targets()
                    for adv_i, a_i in enumerate(adversary_ind):
                        recent_samples = replay_buffer.asyn_sample(config["train"]["batch_size"], to_gpu=USE_CUDA, mode="recent")
                        # if a_i in recv_agent_indices:
                        #     maddpg.update_gh(recent_samples, a_i, logger=logger)
                        maddpg.update_surrogate_policies(recent_samples, agent_obs_with_comm_list, a_i, update_num=3, step_len=0.3, mode="homo", logger=logger, is_debug=config["environment"]["debug_mode"])
                        # maddpg.update_surrogate_policies_GAN(recent_samples, a_i, mode="homo", logger=logger, is_debug=config["environment"]["debug_mode"])
                    maddpg.update_surrogate_target()    

        ep_rews = replay_buffer.get_average_rewards(config["train"]["episode_length"] * config["train"]["n_rollout_threads"])
        std_rews = replay_buffer.get_std_rewards(config["train"]["episode_length"] * config["train"]["n_rollout_threads"])

        ep_rews_comm = replay_buffer_comm.get_average_rewards(config["train"]["episode_length"] * config["train"]["n_rollout_threads"])
        std_rews_comm = replay_buffer_comm.get_std_rewards(config["train"]["episode_length"] * config["train"]["n_rollout_threads"])

        for a_i, (a_ep_rew, a_std_rew) in enumerate(zip(ep_rews, std_rews)):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalar('agent%i/std_episode_rewards' % a_i, a_std_rew, ep_i)
        logger.add_scalar('comm_agent/mean_episode_rewards', ep_rews_comm, ep_i)
        logger.add_scalar('comm_agent/std_episode_rewards', std_rews_comm, ep_i)
        if ep_i > config["train"]["advcomm_n_episodes"]:
            logger.add_scalar('comm_agent/flipped_num', flipped_num, ep_i)
        
        if ep_i % config["train"]["save_interval"] < config["train"]["n_rollout_threads"]:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            os.makedirs(run_dir / 'stats', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
            comm_ddpg.save(run_dir / 'incremental' / ('comm_model_ep%i.pt' % (ep_i + 1)))
            comm_ddpg.save(run_dir / 'comm_model.pt')
            # replay_buffer.generate_prob_map(run_dir / 'stats' / ('rb_ep%i' % (ep_i + 1)), ep_i, adversary_ind)

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    config = config_loader(path="./parameters_training.yaml")  # load model configuration
    seeds = [2]
    adversary_envs = ["random_prey"]
    lrs = [0.0001]
    flip_modes = ["flip"]
    for seed in seeds:
        for lr in lrs:
            for adversary_env in adversary_envs:
                for flip_mode in flip_modes:
                    # INFO: Modify the config
                    config["environment"]["seed"] = seed
                    config["train"]["lr"] = lr
                    config["environment"]["adversary_env"] = adversary_env
                    config["train"]["flip_mode"] = flip_mode   

                    run_adv_comm_offpolicy(config)
