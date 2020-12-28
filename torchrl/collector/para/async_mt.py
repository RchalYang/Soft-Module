
import torch
import copy
import numpy as np

from .base import AsyncParallelCollector
import torch.multiprocessing as mp

import torchrl.policies as policies

from torchrl.env.get_env import *
from torchrl.env.continuous_wrapper import *

from metaworld_utils.meta_env import generate_single_mt_env

from metaworld_utils.meta_env import get_meta_env

from collections import OrderedDict


class AsyncSingleTaskParallelCollector(AsyncParallelCollector):
    def __init__(
            self,
            reset_idx=False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(
            shared_pf, env_info, shared_que, start_barrier, epochs, reset_idx):

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())

            eval_rews = []

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):
                if reset_idx:
                    eval_ob = env_info.env.reset_with_index(idx)
                else:
                    eval_ob = env_info.env.reset()
                rew = 0
                current_success = 0
                while not done:
                    act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()

                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
    
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.env_info.env_cls  = self.env_cls
        self.env_info.env_args = self.env_args

        for i in range(self.worker_nums):
            self.env_info.env_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        # self.eval_start_barrier.wait()
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
            mean_success_rate += worker_rst["success_rate"]

        return {
            'eval_rewards':eval_rews,
            'mean_success_rate': mean_success_rate / self.eval_worker_nums
        }


class AsyncMultiTaskParallelCollectorUniform(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks = list(self.env_cls.keys())
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        task_idx = env_info.env_rank
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)

        embedding_flag = isinstance(pf, policies.EmbeddingGuassianContPolicyBase)

        pf.eval()

        with torch.no_grad():
            if idx_flag:
                idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                if embedding_flag:
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), embedding_input,
                        [task_idx])
                else:
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
                        idx_input)
                act = out["action"]
                # act = act[0]
            else:
                if embedding_flag:
                    # embedding_input = np.zeros(env_info.num_tasks)
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0), embedding_input)
                else:    
                    out = pf.explore(torch.Tensor( ob ).to(env_info.device).unsqueeze(0))
                act = out["action"]


        act = act.detach().cpu().numpy()
        if not env_info.continuous:
            act = act[0]
        
        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        next_ob, reward, done, info = env_info.env.step(act)
        if env_info.train_render:
            env_info.env.render()
        env_info.current_step += 1

        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idxs": [env_info.env_rank],
            "rewards": [reward],
            "terminals": [done]
        }
        if embedding_flag:
            sample_dict["embedding_inputs"] = embedding_input.cpu().numpy()

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.env_rank)

        return next_ob, done, reward, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
        replay_buffer, shared_que,
        start_barrier, epochs, start_epoch, task_name, shared_dict):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        if norm_obs_flag:
            shared_dict[task_name] = {
                "obs_mean": env_info.env._obs_mean,
                "obs_var": env_info.env._obs_var
            }
            # print("Put", task_name)

        c_ob = {
            "ob": env_info.env.reset()
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            for key in shared_funcs:
                local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_epoch_reward = 0    

            for _ in range(env_info.epoch_frames):
                # print(env_info.epoch_frames)
                next_ob, done, reward, _ = cls.take_actions(local_funcs, env_info, c_ob, replay_buffer )
                c_ob["ob"] = next_ob
                train_rew += reward
                train_epoch_reward += reward
                if done:
                    train_rews.append(train_rew)
                    train_rew = 0

            if norm_obs_flag:
                shared_dict[task_name] = {
                    "obs_mean": env_info.env._obs_mean,
                    "obs_var": env_info.env._obs_var
                }
                # print("Put", task_name)

            shared_que.put({
                'train_rewards':train_rews,
                'train_epoch_reward':train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict):

        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.EmbeddingGuassianContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'success_rate': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            # print("Get", task_name)
            if norm_obs_flag:
                env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = shared_dict[task_name]["obs_var"]
                # print(env_info.env._obs_mean)
                #  = {
                #     "obs_mean": env_info.env._obs_mean,
                #     "obs_var": env_info.env._obs_var
                # }

            eval_rews = []  

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.env_rank
                current_success = 0
                while not done:

                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),
                                embedding_input, [task_idx] )
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    else:
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input)
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))

                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes,
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)


        self.shared_dict = self.manager.dict()

        assert self.worker_nums == self.env.num_tasks
        # task_cls, task_args, env_params
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }
        
        tasks = list(self.env_cls.keys())

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]
            
            self.env_info.env_rank = i
            
            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            if "start_epoch" in self.env_info.env_args["task_args"]:
                start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            else:
                start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.shared_funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.train_epochs, start_epoch, task, self.shared_dict))
            p.start()
            self.workers.append(p)
            # i += 1

        assert self.eval_worker_nums == self.env.num_tasks
        
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task, self.shared_dict))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                mean_success_rate += worker_rst["success_rate"]
                tasks_result.append((worker_rst["task_name"], worker_rst["success_rate"], np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, success_rate, eval_rewards in tasks_result:
            dic[task_name+"_success_rate"] = success_rate
            dic[task_name+"_eval_rewards"] = eval_rewards
            # if self.tasks_progress[self.tasks_mapping[task_name]] is None:
            #     self.tasks_progress[self.tasks_mapping[task_name]] = success_rate
            # else:
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)
            self.tasks_progress[self.tasks_mapping[task_name]] += \
                self.progress_alpha * success_rate

        dic['eval_rewards']      = eval_rews
        dic['mean_success_rate'] = mean_success_rate / active_task_counts

        return dic


    def train_one_epoch(self):
        train_rews = []
        train_epoch_reward = 0

        for key in self.shared_funcs:
            self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())
        
        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            if worker_rst["train_rewards"] is not None:
                train_rews += worker_rst["train_rewards"]
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums

        return {
            'train_rewards':train_rews,
            'train_epoch_reward':train_epoch_reward
        }


class AsyncMultiTaskParallelCollectorUniformImitation(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks = list(self.env_cls.keys())
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha

    @staticmethod
    def eval_worker_process(shared_pf, 
        env_info, shared_que, start_barrier, epochs, start_epoch, task_name):

        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies.EmbeddingGuassianContPolicyBase) or isinstance(pf,
            policies.EmbeddingDetContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'success_rate': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            eval_rews = []  

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.env_rank
                current_success = 0
                while not done:

                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0),
                                embedding_input, [task_idx] )
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), idx_input )
                    else:
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            # embedding_input = torch.cat([torch.Tensor(env_info.env.goal.copy()), embedding_input])
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), embedding_input)
                        else:
                            act = pf.eval_act( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))

                    eval_ob, r, done, info = env_info.env.step( act )
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes,
                'task_name': task_name
            })


    def start_worker(self):                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        # task_cls, task_args, env_params
        tasks = list(self.env_cls.keys())
        
        assert self.worker_nums == 0
        assert self.eval_worker_nums == self.env.num_tasks
        
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
                # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.eval_epochs, start_epoch, task))
            eval_p.start()
            self.eval_workers.append(eval_p)


    def eval_one_epoch(self):
        
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                mean_success_rate += worker_rst["success_rate"]
                tasks_result.append((worker_rst["task_name"], worker_rst["success_rate"], np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, success_rate, eval_rewards in tasks_result:
            dic[task_name+"_success_rate"] = success_rate
            dic[task_name+"_eval_rewards"] = eval_rewards
            # if self.tasks_progress[self.tasks_mapping[task_name]] is None:
            #     self.tasks_progress[self.tasks_mapping[task_name]] = success_rate
            # else:
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)
            self.tasks_progress[self.tasks_mapping[task_name]] += \
                self.progress_alpha * success_rate

        dic['eval_rewards']      = eval_rews
        dic['mean_success_rate'] = mean_success_rate / active_task_counts

        return dic
