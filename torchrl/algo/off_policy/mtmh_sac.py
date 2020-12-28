from .twin_sac_q import TwinSACQ
from .mt_sac import MTSAC
import copy
import torch
import torchrl.algo.utils as atu
import numpy as np
import torch.nn.functional as F

class MTMHSAC(MTSAC):
    ## Multi Task Multi Head SAC (Input Processed)
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.head_idx = list(range(self.task_nums))
        self.sample_key = ["obs", "next_obs", "acts", "rewards", "task_idxs",
                           "terminals"]
        if self.pf_flag:
            self.sample_key.append("embedding_inputs")

    def update(self, batch):
        self.training_update_num += 1
        
        obs       = batch['obs']
        actions   = batch['acts']
        next_obs  = batch['next_obs']
        rewards   = batch['rewards']
        terminals = batch['terminals']

        # For Task
        task_idx    = batch['task_idxs']
        # task_onehot = batch['task_onehot']
        if self.pf_flag:
            embedding_inputs = batch["embedding_inputs"]

        rewards = torch.Tensor(rewards).to( self.device )
        terminals = torch.Tensor(terminals).to( self.device )
        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        next_obs = torch.Tensor(next_obs).to( self.device )
        # For Task
        task_idx    = torch.Tensor(task_idx).to( self.device ).long()
        # task_onehot = torch.Tensor(task_onehot).to( self.device )

        if self.pf_flag:
            embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)

        """
        Policy operations.
        """
        if self.pf_flag:
            sample_info = self.pf.explore(obs, embedding_inputs,
                self.head_idx, return_log_probs=True )
        else:
            sample_info = self.pf.explore(obs, self.head_idx, return_log_probs=True )

        mean_list        = sample_info["mean"]
        log_std_list     = sample_info["log_std"]
        new_actions_list = sample_info["action"]
        log_probs_list   = sample_info["log_prob"]
        # ent_list         = sample_info["ent"]

        means = atu.unsqe_cat_gather(mean_list, task_idx, dim = 1)

        log_stds = atu.unsqe_cat_gather(log_std_list, task_idx, dim = 1)

        new_actions = atu.unsqe_cat_gather(new_actions_list, task_idx, dim = 1)

        # log_probs = torch.cat
        log_probs = atu.unsqe_cat_gather(log_probs_list, task_idx, dim = 1)

        if self.pf_flag:
            q1_pred_list = self.qf1([obs, actions], embedding_inputs, self.head_idx)
            q2_pred_list = self.qf2([obs, actions], embedding_inputs, self.head_idx)
        else:
            q1_pred_list = self.qf1([obs, actions], self.head_idx)
            q2_pred_list = self.qf2([obs, actions], self.head_idx)
            
        q1_preds = atu.unsqe_cat_gather(q1_pred_list, task_idx, dim = 1)
        q2_preds = atu.unsqe_cat_gather(q2_pred_list, task_idx, dim = 1)

        reweight_coeff = 1
        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            batch_size = log_probs.shape[0]
            log_alphas = (self.log_alpha.unsqueeze(0)).expand((batch_size, self.task_nums))
            log_alphas = log_alphas.gather(1, task_idx)

            alpha_loss = -(log_alphas * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # alpha = self.log_alpha.exp()
            alphas = (self.log_alpha.exp().detach()).unsqueeze(0).expand((batch_size, self.task_nums))
            alphas = alphas.gather(1, task_idx)
            if self.temp_reweight:
                softmax_temp = F.softmax(-self.log_alpha.detach())
                reweight_coeff = softmax_temp.unsqueeze(0).expand((batch_size, self.task_nums))
                reweight_coeff = reweight_coeff.gather(1, task_idx)
        else:
            alphas = 1
            alpha_loss = 0

        progress_weight = 1
        if self.progress_reweight:
            progress_weight = torch.Tensor(self.collector.task_progress)
            progress_weight = progress_weight.unsqueeze(0).expand((batch_size, self.task_nums))
            progress_weight = progress_weight.gather(1, task_idx)
        reweight_coeff = reweight_coeff * progress_weight

        with torch.no_grad():
            if self.pf_flag:
                target_sample_info = self.pf.explore(next_obs, embedding_inputs,
                    self.head_idx, return_log_probs=True )
            else:
                target_sample_info = self.pf.explore(next_obs, self.head_idx, return_log_probs=True )

            target_actions_list   = target_sample_info["action"]
            target_actions = atu.unsqe_cat_gather(target_actions_list, task_idx, dim = 1)

            target_log_probs_list = target_sample_info["log_prob"]
            target_log_probs = atu.unsqe_cat_gather(target_log_probs_list, task_idx, dim = 1)

            if self.pf_flag:
                target_q1_pred_list = self.target_qf1([next_obs, target_actions],
                    embedding_inputs, self.head_idx)    
                target_q2_pred_list = self.target_qf2([next_obs, target_actions],
                    embedding_inputs, self.head_idx)
            else:
                target_q1_pred_list = self.target_qf1([next_obs, target_actions], self.head_idx)    
                target_q2_pred_list = self.target_qf2([next_obs, target_actions], self.head_idx)

            target_q1_pred = atu.unsqe_cat_gather(target_q1_pred_list, task_idx, dim = 1)
            target_q2_pred = atu.unsqe_cat_gather(target_q2_pred_list, task_idx, dim = 1)

            min_target_q = torch.min(target_q1_pred, target_q2_pred)
            target_v_values = min_target_q - alphas * target_log_probs

        """
        QF Loss
        """
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # There is no actual terminate in meta-world -> just filter all time_limit terminal
        q_target = rewards + self.discount * target_v_values

        qf1_loss = (reweight_coeff * ((q1_preds - q_target.detach()) ** 2)).mean()
        qf2_loss = (reweight_coeff * ((q2_preds - q_target.detach()) ** 2)).mean()

        # """
        # VF Loss
        # """
        if self.pf_flag:
            q1_new_actions_list = self.qf1([obs, new_actions],
                embedding_inputs, self.head_idx)
            q2_new_actions_list = self.qf2([obs, new_actions],
                embedding_inputs, self.head_idx)
        else:
            q1_new_actions_list = self.qf1([obs, new_actions], self.head_idx)
            q2_new_actions_list = self.qf2([obs, new_actions], self.head_idx)

        q1_new_actions = atu.unsqe_cat_gather(q1_new_actions_list, task_idx, dim = 1)
        q2_new_actions = atu.unsqe_cat_gather(q2_new_actions_list, task_idx, dim = 1)

        q_new_actions = torch.min(
            q1_new_actions,
            q2_new_actions
        ) 

        """
        Policy Loss
        """
        if not self.reparameterization:
            raise NotImplementedError
        else:
            # policy_loss = ( alphas * log_probs - q_new_actions).mean()
            assert log_probs.shape == q_new_actions.shape
            policy_loss = (reweight_coeff * ( alphas * log_probs - q_new_actions)).mean()

        std_reg_loss = self.policy_std_reg_weight * (log_stds**2).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (means**2).mean()

        policy_loss += std_reg_loss + mean_reg_loss

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()

        if self.automatic_entropy_tuning:
            for i in range(self.task_nums):
                info["alpha_{}".format(i)] = self.log_alpha[i].exp().item()
            info["Alpha_loss"] = alpha_loss.item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qf1_loss'] = qf1_loss.item()
        info['Training/qf2_loss'] = qf2_loss.item()

        info['log_std/mean'] = log_stds.mean().item()
        info['log_std/std'] = log_stds.std().item()
        info['log_std/max'] = log_stds.max().item()
        info['log_std/min'] = log_stds.min().item()

        info['log_probs/mean'] = log_probs.mean().item()
        info['log_probs/std'] = log_probs.std().item()
        info['log_probs/max'] = log_probs.max().item()
        info['log_probs/min'] = log_probs.min().item()

        info['mean/mean'] = means.mean().item()
        info['mean/std'] = means.std().item()
        info['mean/max'] = means.max().item()
        info['mean/min'] = means.min().item()

        return info

    def update_per_epoch(self):
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    reshape=True)
            infos = self.update(batch)
            self.logger.add_update_info(infos)
