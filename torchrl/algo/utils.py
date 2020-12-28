import torch
import numpy as np


def quantile_regression_loss(coefficient, source, target):
    diff = target.unsqueeze(-1) - source.unsqueeze(1)
    loss = huber(diff) * (coefficient - (diff.detach() < 0).float()).abs()
    loss = loss.mean()
    return loss


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def unsqe_cat_gather(tensor_list, idx, dim = 1 ):
    tensor_list = [tensor.unsqueeze(dim) for tensor in tensor_list]
    tensors = torch.cat(tensor_list, dim = dim)

    target_shape = list(tensors.shape)
    target_shape[dim] = 1

    view_shape = list(idx.shape) + [1] * (len(target_shape) - len(idx.shape))
    idx = idx.view(view_shape)
    idx = idx.expand(tuple(target_shape))
    tensors = tensors.gather(dim, idx).squeeze(dim)
    return tensors


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   
