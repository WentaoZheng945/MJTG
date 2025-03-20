import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor_2(nn.Module):
    """Generator of GAN"""
    def __init__(self, d_z, d_model, layer_num=3):
        super(Actor_2, self).__init__()
        self.d_z = d_z
        self.d_model = d_model
        self.layer_num = layer_num
        layer_list = []
        for i in range(layer_num):
            if i == 0:
                input_dim = d_z
                layer_list.append(nn.Linear(input_dim, self.d_model))
            else:
                layer_list.append(nn.Linear(self.d_model, self.d_model))
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Linear(self.d_model, self.d_z * 2))

        self.fw_layer = nn.Sequential(*layer_list)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        out = self.fw_layer(x)
        input_gate, dz = out.chunk(2, dim=-1)
        gate_value = self.gate(input_gate)
        new_z = (1 - gate_value) * original_x + gate_value * dz
        return new_z


class Critic_2(nn.Module):
    '''Discriminator of GAN'''
    def __init__(self, d_z, d_model, layer_num=3, num_output=1):
        super(Critic_2, self).__init__()
        self.d_z = d_z
        self.d_model = d_model
        layer_list = []
        for i in range(layer_num):
            if i == 0:
                input_dim = d_z
                layer_list.append(nn.Linear(input_dim, d_model))
            else:
                layer_list.append(nn.Linear(d_model, d_model))
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Linear(d_model, num_output))
        self.fw_layer = nn.Sequential(*layer_list)

    def forward(self, x):
        out = self.fw_layer(x)
        return out