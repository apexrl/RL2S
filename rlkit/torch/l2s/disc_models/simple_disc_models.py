import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch import pytorch_util as ptu
from torch.nn.utils import spectral_norm
from rlkit.torch.l2s.disc_models.snlayers import SNLinear, SNConv2d
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer


def spectal_wrapper(model,use_sn=False):
    if use_sn:
        return spectral_norm(model)
    else:
        return model

class MLPDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=False,
        use_sn=True,
        if_clamp=False,
        clamp_magnitude=10.0
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.if_clamp = if_clamp
        self.clamp_magnitude = clamp_magnitude
        self.use_sn = use_sn   

         
        self.mod_list = nn.ModuleList([spectal_wrapper(nn.Linear(input_dim, hid_dim),self.use_sn)])
        if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(spectal_wrapper(nn.Linear(hid_dim, hid_dim),self.use_sn))
            if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())
        
        self.mod_list.append(spectal_wrapper(nn.Linear(hid_dim, 1),self.use_sn))
        self.model = nn.Sequential(*self.mod_list)


    def forward(self,batch):
        output = self.model(batch)
        if self.if_clamp:
            output_clamp = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
            return output,output_clamp
        return output,output