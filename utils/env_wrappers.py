"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from environment.vec_env import VecEnv

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns] # self.envs: the returning from make_env, that is, MultiAgentEnv class
        self.agent_types = []
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            for a in env.agents:
                if a.adversary:
                    self.agent_types.append('adversary')
                elif a.prey:
                    self.agent_types.append('prey')
                else:
                    self.agent_types.append('capturer')
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions # [[Env 1: agent 1 action, agent 2 action,...agent Nr action],[Env 2: ],...,[Env Ne: ]]

    def step_wait(self):
        # update one environment with the actions in it, env: MultiAgentEnv
        results = [env._step(a) for (a,env) in zip(self.actions, self.envs)] # results: obs_n (16/14)*Nr, reward_n (1)*Nr, done_n (False array), info_n (nothing)
        obs, rews, dones, infos = map(np.array, zip(*results)) # map: assign function handle to the iterator
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos # (1,4,16/14), (1,4), (1,4), array([{'n': [{}, {}, {}, {}]}], dtype=object)

    def reset(self, seed):        
        results = [env._reset(env_seed = None) for env in self.envs]
        return np.array(results)

    # def reset(self, seed):        
    #     results = [np.array(env._reset(env_seed = None)).astype(np.object) for env in self.envs]
    #     return results

    def get_obs(self): # get observation for all agents
        results = [env._get_obs_for_all() for env in self.envs]
        return np.array(results)

    def translate(self, agent1_ind, agent2_ind):
        obs_n = []
        for env in self.envs:
            obs_n.append(env._translate(agent1_ind, agent2_ind))
        return np.array(obs_n)

    def translate_one(self, agent1_ind, agent2_ind):
        obs_n = []
        for env in self.envs:
            obs_n.append(env._translate_one(agent1_ind, agent2_ind))
        return np.array(obs_n)

    def render(self, render_option = "human"):
        env_to_render = self.envs[0]
        ret = env_to_render._render(render_option)
        return ret
        
    def close(self):
        return