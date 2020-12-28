from .continuous_wrapper import *
from .base_wrapper import *
import os
import gym
import mujoco_py
import xml.etree.ElementTree as ET


def wrap_continuous_env(env, obs_norm, reward_scale):
    env = RewardShift(env, reward_scale)
    if obs_norm:
        return NormObs(env)
    return env


def get_env( env_id, env_param ):
    # env = gym.make(env_id)
    # if "customize" in env_param:
    #     env = customized_mujoco(env, env_param["customize"])
    #     del env_param["customize"]
    # env = BaseWrapper(env)
    env = BaseWrapper(gym.make(env_id))
    if "rew_norm" in env_param:
        env = NormRet(env, **env_param["rew_norm"])
        del env_param["rew_norm"]

    ob_space = env.observation_space
    env = wrap_continuous_env(env, **env_param)

    if str(env.__class__.__name__).find('TimeLimit') >= 0:
        env = TimeLimitAugment(env)

    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Box):
        return NormAct(env)
    return env
