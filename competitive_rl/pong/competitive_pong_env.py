import random

import numpy as np

from competitive_rl.pong.builtin_policies import get_compute_action_function, \
    get_builtin_agent_names


class TournamentEnvWrapper:
    def __init__(self, env, num_envs):
        self.env = env
        self.agents = {
            agent_name: get_compute_action_function(agent_name, num_envs)
            for agent_name in get_builtin_agent_names()
            if agent_name != "ALPHA_PONG"
        }
        self.agent_names = list(self.agents)
        self.prev_opponent_obs = None
        self.current_agent_name = "RULE_BASED"
        self.current_agent = self.agents[self.current_agent_name]
        self.observation_space = env.observation_space[0]
        self.action_space = env.action_space[0]
        self.num_envs = num_envs

    def get_agent_names(self):
        return self.agent_names

    def reset_opponent(self, agent_name=None):
        if agent_name is None:
            self.current_agent_name = random.choice(self.agent_names)
        else:
            assert agent_name in self.agent_names, self.agent_names
            self.current_agent_name = agent_name
        self.current_agent = self.agents[self.current_agent_name]

    def step(self, action):
        tuple_action = np.stack([
            np.asarray(action).reshape(-1),
            np.asarray(self.current_agent(self.prev_opponent_obs)).reshape(-1)
        ], axis=1)
        result = self.env.step(tuple_action)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
            done = np.logical_or(terminated, truncated)
        else:
            obs, rew, done, info = result
        if isinstance(obs, (tuple, list)) and len(obs) > 1:
            self.prev_opponent_obs = obs[1]
            obs = obs[0]
        else:
            self.prev_opponent_obs = obs
        if done.ndim == 2:
            done = done[:, 0]
        return obs, rew[:, 0].reshape(-1, 1), done.reshape(-1, 1), info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, (tuple, list)) and len(obs) > 1:
            self.prev_opponent_obs = obs[1]
            obs = obs[0]
        else:
            self.prev_opponent_obs = obs
        return obs

    def seed(self, s):
        self.env.seed(s)

    def close(self):
        self.env.close()
