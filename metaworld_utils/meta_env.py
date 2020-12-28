
import gym
from gym import Wrapper
from gym.spaces import Box
import numpy as np
from metaworld.envs.mujoco.sawyer_xyz import *
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.core.serializable import Serializable
import sys
sys.path.append("../..")
from torchrl.env.continuous_wrapper import *
from torchrl.env.get_env import wrap_continuous_env


class SingleWrapper(Wrapper):
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self._env.reset()

    def seed(self, se):
        self._env.seed(se)

    def reset_with_index(self, task_idx):
        return self._env.reset()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return self._env.render(mode=mode, **kwargs)

    def close(self):
        self._env.close()


class MTEnv(MultiClassMultiTaskEnv):
    def __init__(self,
                 task_env_cls_dict,
                 task_args_kwargs,
                 sample_all=True,
                 sample_goals=False,
                 obs_type='plain',
                 repeat_times=1,
                 ):
        Serializable.quick_init(self, locals())
        super().__init__(
            task_env_cls_dict,
            task_args_kwargs,
            sample_all,
            sample_goals,
            obs_type)

        self.train_mode = True
        self.repeat_times = repeat_times

    def reset(self, **kwargs):
        if self.train_mode:
            sample_task = np.random.randint(0, self.num_tasks)
            self.set_task(sample_task)
        return super().reset(**kwargs)

    def reset_with_index(self, task_idx, **kwargs):
        self.set_task(task_idx)
        return super().reset(**kwargs)

    def train(self):
        self.train_mode = True

    def test(self):
        self.train_mode = False

    def render(self, mode='human'):
        return super().render(mode=mode)

    @property
    def observation_space(self):
        if self._obs_type == 'plain':
            return self._task_envs[self.observation_space_index].observation_space
        else:
            plain_high = self._task_envs[self.observation_space_index].observation_space.high
            plain_low = self._task_envs[self.observation_space_index].observation_space.low
            goal_high = self.active_env.goal_space.high
            goal_low = self.active_env.goal_space.low
            if self._obs_type == 'with_goal':
                return Box(
                    high=np.concatenate([plain_high, goal_high] + [goal_high] * (self.repeat_times -1) ),
                    low=np.concatenate([plain_low, goal_low] + [goal_low] * (self.repeat_times -1 )))
            elif self._obs_type == 'with_goal_id' and self._fully_discretized:
                goal_id_low = np.zeros(shape=(self._n_discrete_goals * self.repeat_times,))
                goal_id_high = np.ones(shape=(self._n_discrete_goals * self.repeat_times,))
                return Box(
                    high=np.concatenate([plain_high, goal_id_low,]),
                    low=np.concatenate([plain_low, goal_id_high,]))
            elif self._obs_type == 'with_goal_and_id' and self._fully_discretized:
                goal_id_low = np.zeros(shape=(self._n_discrete_goals,))
                goal_id_high = np.ones(shape=(self._n_discrete_goals,))
                return Box(
                    high=np.concatenate([plain_high, goal_id_low, goal_high] + [goal_id_low, goal_high] * (self.repeat_times - 1) ),
                    low=np.concatenate([plain_low, goal_id_high, goal_low] + [goal_id_high, goal_low] * (self.repeat_times - 1) ))
            else:
                raise NotImplementedError

    def _augment_observation(self, obs):
        # optionally zero-pad observation
        if np.prod(obs.shape) < self._max_plain_dim:
            zeros = np.zeros(
                shape=(self._max_plain_dim - np.prod(obs.shape),)
            )
            obs = np.concatenate([obs, zeros])

        # augment the observation based on obs_type:
        if self._obs_type == 'with_goal_id' or self._obs_type == 'with_goal_and_id':

            aug_ob = []
            if self._obs_type == 'with_goal_and_id':
                aug_ob.append(self.active_env._state_goal)
            # if self._obs_type == 'with_goal_and_id':
            #     obs = np.concatenate([obs, self.active_env._state_goal])
            task_id = self._env_discrete_index[self._task_names[self.active_task]] + (self.active_env.active_discrete_goal or 0)
            task_onehot = np.zeros(shape=(self._n_discrete_goals,), dtype=np.float32)
            task_onehot[task_id] = 1.
            aug_ob.append(task_onehot)

            obs = np.concatenate([obs] + aug_ob * self.repeat_times)

        elif self._obs_type == 'with_goal':
            obs = np.concatenate([obs] + [self.active_env._state_goal] * self.repeat_times )
        return obs


def generate_single_task_env(env_id, kwargs):
    env = globals()[env_id](**kwargs)
    env = SingleWrapper(env)
    return env


def generate_mt_env(cls_dict, args_kwargs, **kwargs):
    copy_kwargs = kwargs.copy()
    if "random_init" in copy_kwargs:
        del copy_kwargs["random_init"]
    env = MTEnv(
        task_env_cls_dict=cls_dict,
        task_args_kwargs=args_kwargs,
        **copy_kwargs
    )
    # Set to discretized since the env is actually not used
    env._sample_goals = False
    env._fully_discretized = True

    goals_dict = {
        t: [e.goal.copy()]
        for t, e in zip(env._task_names, env._task_envs)
    }
    env.discretize_goal_space(goals_dict)
    return env


def generate_single_mt_env(task_cls, task_args, env_rank, num_tasks,
                           max_obs_dim, env_params, meta_env_params):

    env = task_cls(*task_args['args'], **task_args["kwargs"])
    env.discretize_goal_space(env.goal.copy())
    if "sampled_index" in meta_env_params:
        del meta_env_params["sampled_index"]
    env = AugObs(env, env_rank, num_tasks, max_obs_dim, meta_env_params)
    env = wrap_continuous_env(env, **env_params)

    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        env = NormAct(env)
    return env


def generate_mt10_env(mt_param):
    from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS

    if "random_init" in mt_param:
        for key in EASY_MODE_ARGS_KWARGS:
            EASY_MODE_ARGS_KWARGS[key]["kwargs"]["random_init"]=True

    return generate_mt_env(EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS, **mt_param), \
        EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS


def generate_mt50_env(mt_param):
    from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
    cls_dict = {}
    args_kwargs = {}
    for k in HARD_MODE_CLS_DICT.keys():
        for task in HARD_MODE_CLS_DICT[k].keys():
            cls_dict[task] = HARD_MODE_CLS_DICT[k][task]
            args_kwargs[task] = HARD_MODE_ARGS_KWARGS[k][task]

    if "random_init" in mt_param:
        for key in args_kwargs:
            args_kwargs[key]["kwargs"]["random_init"]=mt_param["random_init"]

    return generate_mt_env(cls_dict, args_kwargs, **mt_param), \
        cls_dict, args_kwargs


def get_meta_env(env_id, env_param, mt_param, return_dicts=True):
    cls_dicts = None
    args_kwargs = None
    if env_id == "mt10":
        env, cls_dicts, args_kwargs = generate_mt10_env(mt_param)
    elif env_id == "mt50":
        env, cls_dicts, args_kwargs = generate_mt50_env(mt_param)
    else:
        env = generate_single_task_env(env_id, mt_param)

    env = wrap_continuous_env(env, **env_param)

    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        env = NormAct(env)
    if env_id == "mt10" or env_id == "mt50":
        env.num_tasks = len(cls_dicts)
    else:
        env.num_tasks = 1

    if cls_dicts is not None and return_dicts is True:
        return env, cls_dicts, args_kwargs
    else:
        return env
