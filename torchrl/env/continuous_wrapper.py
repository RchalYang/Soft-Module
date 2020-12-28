import gym
import numpy as np

from .base_wrapper import BaseWrapper
from gym.spaces import Box


class AugObs(gym.ObservationWrapper, BaseWrapper):
    def __init__(self, env, env_rank, num_tasks, max_obs_dim, meta_env_params):
        super(AugObs, self).__init__(env)
        self.env_rank = env_rank
        self.num_tasks = num_tasks
        self.task_onehot = np.zeros(shape=(num_tasks,), dtype=np.float32)
        self.task_onehot[env_rank] = 1.
        self.max_obs_dim = max_obs_dim
        self.obs_type = meta_env_params["obs_type"]
        self.obs_dim = np.prod(env.observation_space.shape)

        if self.obs_type == "with_goal_and_id":
            self.obs_dim += num_tasks
            self.obs_dim += np.prod(env._state_goal.shape)
        elif self.obs_type == "with_goal":
            self.obs_dim += np.prod(env._state_goal.shape)
        elif self.obs_type == "with_goal_id":
            self.obs_dim += num_tasks

        if self.obs_dim < self.max_obs_dim:
            self.pedding = np.zeros(self.max_obs_dim - self.obs_dim)

        self.repeat_times = meta_env_params["repeat_times"] \
            if "repeat_times" in meta_env_params else 1

        # self.set_observation_space()

        # if self.obs_type == 'plain':
        #     self.observation_space = self._wrapped_env.observation_space
        # else:
        #     plain_high = self._wrapped_env.observation_space.high
        #     plain_low = self._wrapped_env.observation_space.low
        #     goal_high = self._wrapped_env.goal_space.high
        #     goal_low = self._wrapped_env.goal_space.low
        #     if self.obs_type == 'with_goal':
        #         self.observation_space = Box(
        #             high=np.concatenate([plain_high, goal_high] + [goal_high] * (self.repeat_times -1) ),
        #             low=np.concatenate([plain_low, goal_low] + [goal_low] * (self.repeat_times -1 )))
        #     elif self.obs_type == 'with_goal_id' and self._fully_discretized:
        #         goal_id_low = np.zeros(shape=(self._n_discrete_goals * self.repeat_times,))
        #         goal_id_high = np.ones(shape=(self._n_discrete_goals * self.repeat_times,))
        #         self.observation_space = Box(
        #             high=np.concatenate([plain_high, goal_id_low,]),
        #             low=np.concatenate([plain_low, goal_id_high,]))
        #     elif self.obs_type == 'with_goal_and_id' and self._fully_discretized:
        #         goal_id_low = np.zeros(shape=(self._n_discrete_goals,))
        #         goal_id_high = np.ones(shape=(self._n_discrete_goals,))
        #         self.observation_space = Box(
        #             high=np.concatenate([plain_high, goal_id_low, goal_high] + [goal_id_low, goal_high] * (self.repeat_times - 1) ),
        #             low=np.concatenate([plain_low, goal_id_high, goal_low] + [goal_id_high, goal_low] * (self.repeat_times - 1) ))
        #     else:
        #         raise NotImplementedError


    def observation(self, observation):

        if self.obs_type == "with_goal_and_id":
            aug_ob = np.concatenate([self._wrapped_env._state_goal,
                                     self.task_onehot])
        elif self.obs_type == "with_goal":
            aug_ob = self._wrapped_env._state_goal
        elif self.obs_type == "with_goal_id":
            aug_ob = self.task_onehot
        elif self.obs_type == "plain":
            aug_ob = []

        aug_ob = np.concatenate([aug_ob] * self.repeat_times)
        if self.obs_dim < self.max_obs_dim:
            observation = np.concatenate([observation, self.pedding])
        observation = np.concatenate([observation, aug_ob])
        return observation

    # # @property
    # def set_observation_space(self):
    #     if self._obs_type == 'plain':
    #         self.observation_space = self._wrapped_env.observation_space
    #     else:
    #         plain_high = self._wrapped_env.observation_space.high
    #         plain_low = self._wrapped_env.observation_space.low
    #         goal_high = self._wrapped_env.goal_space.high
    #         goal_low = self._wrapped_env.goal_space.low
    #         if self._obs_type == 'with_goal':
    #             self.observation_space = Box(
    #                 high=np.concatenate([plain_high, goal_high] + [goal_high] * (self.repeat_times -1) ),
    #                 low=np.concatenate([plain_low, goal_low] + [goal_low] * (self.repeat_times -1 )))
    #         elif self._obs_type == 'with_goal_id' and self._fully_discretized:
    #             goal_id_low = np.zeros(shape=(self._n_discrete_goals * self.repeat_times,))
    #             goal_id_high = np.ones(shape=(self._n_discrete_goals * self.repeat_times,))
    #             self.observation_space = Box(
    #                 high=np.concatenate([plain_high, goal_id_low,]),
    #                 low=np.concatenate([plain_low, goal_id_high,]))
    #         elif self._obs_type == 'with_goal_and_id' and self._fully_discretized:
    #             goal_id_low = np.zeros(shape=(self._n_discrete_goals,))
    #             goal_id_high = np.ones(shape=(self._n_discrete_goals,))
    #             self.observation_space = Box(
    #                 high=np.concatenate([plain_high, goal_id_low, goal_high] + [goal_id_low, goal_high] * (self.repeat_times - 1) ),
    #                 low=np.concatenate([plain_low, goal_id_high, goal_low] + [goal_id_high, goal_low] * (self.repeat_times - 1) ))
    #         else:
    #             raise NotImplementedError


class NormObs(gym.ObservationWrapper, BaseWrapper):
    """
    Normalized Observation => Optional, Use Momentum
    """
    def __init__( self, env, obs_alpha = 0.001 ):
        super(NormObs,self).__init__(env)
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape[0])
        self._obs_var = np.ones(env.observation_space.shape[0])

# Check Trajectory is ended by time limit or not
class TimeLimitAugment(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['time_limit'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NormAct(gym.ActionWrapper, BaseWrapper):
    """
    Normalized Action      => [ -1, 1 ]
    """
    def __init__(self, env):
        super(NormAct, self).__init__(env)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub)

    def action(self, action):
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(scaled_action, lb, ub)
