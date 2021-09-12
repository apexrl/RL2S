import numpy as np
import pickle
from rlkit.torch.core import np_to_pytorch_batch
import torch
import pickle as pkl
import joblib
from matplotlib import pyplot as plt


def compute_dataset_q(dataset,trained_q,batch_size=256,file_path=""):
    log_value = []
    with torch.no_grad():
        keys = ['observations','actions']
        data_size = dataset._buffer_size
        iter_nums = int(data_size/batch_size)
        last_index = 0
        for i in range(iter_nums):
            batch = dataset.random_batch(batch_size, keys=keys)
            batch = np_to_pytorch_batch(batch)
            obs = batch['observations']
            acts = batch['actions']
            sa_input = torch.cat([obs, acts], dim=1) 
            q_value = trained_q(sa_input).cpu().numpy()
            q_value = np.squeeze(q_value)
            log_value.extend(list(q_value))
    joblib.dump(log_value,file_path,compress=3)
    print("information write down")

def clip_gradient(optimizer, grad_clip=0.5):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# def draw_pic(file_path):
