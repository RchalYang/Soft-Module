

import torch
import copy
import numpy as np

from .base import ParallelCollector
import torch.multiprocessing as mp

import torchrl.policies as policies

class SingleTaskParallelCollectorBase(ParallelCollector):

    def __init__(self, 
            reset_idx = False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(shared_pf, 
        env_info, shared_que, start_barrier, terminate_mark, reset_idx):

        pf = copy.deepcopy(shared_pf)
        idx_flag = isinstance(pf, policies.MultiHeadGuassianContPolicy)

        env_info.env.eval()
        env_info.env._reward_scale = 1

        while True:
            start_barrier.wait()
            if terminate_mark.value == 1:
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
                    # act = pf.eval( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
                    if idx_flag:
                        act = pf.eval( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0), [task_idx] )
                    else:
                        act = pf.eval( torch.Tensor( eval_ob ).to(env_info.device).unsqueeze(0))
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
        self.shared_que = self.manager.Queue()
        self.start_barrier = mp.Barrier(self.worker_nums+1)
        self.terminate_mark = mp.Value( 'i', 0 )
                
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue()
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums+1)

        for i in range(self.worker_nums):
            self.env_info.env_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=( self.__class__, self.funcs,
                    self.env_info, self.replay_buffer, 
                    self.shared_que, self.start_barrier,
                    self.terminate_mark))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.pf,
                    self.env_info, self.eval_shared_que, self.eval_start_barrier,
                    self.terminate_mark, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        self.eval_start_barrier.wait()
        eval_rews = []
        mean_success_rate = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
            mean_success_rate += worker_rst["success_rate"]
        
        return {
            'eval_rewards':eval_rews,
            'mean_success_rate': mean_success_rate / self.eval_worker_nums
        }
    
    