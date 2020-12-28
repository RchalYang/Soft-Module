import torch
import numpy as np

from .base import BaseCollector

class MultiTaskCollectorBase(BaseCollector):

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer):

        pf = funcs["pf"]
        ob = ob_info["ob"]
        # idx = ob_info["task_index"]
        task_idx = env_info.env.active_task
        out = pf.explore( torch.Tensor( ob ).to(env_info.device).unsqueeze(0),
            [task_idx])
        act = out["action"]
        act = act[0]
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
            "obs":ob,
            "next_obs": next_ob,
            "acts": act,
            "task_idx": task_idx,
            "rewards": [reward],
            "terminals": [done]
        }

        if done or env_info.current_step >= env_info.max_episode_frames:
            next_ob = env_info.env.reset()
            env_info.finish_episode()
            env_info.start_episode() # reset current_step

        replay_buffer.add_sample( sample_dict, env_info.env_rank)

        return next_ob, done, reward, info
    
