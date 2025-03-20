import os
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib as mpl
mpl.use('Agg')


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mse_loss(x, x_recon):
    """
    cal MSE
    """
    return torch.nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)


def get_velocity(velocity, position, rate):
    for i in range(position.shape[0]):
        for j in range(position.shape[1]):
            if j == 0:
                for k in range(position.shape[2]):
                    velocity[i, j, k] = (position[i, j + 2, k] - position[i, j, k]) / float(2 * rate)
            elif j == position.shape[1] - 1:
                for k in range(position.shape[2]):
                    velocity[i, j, k] = (position[i, j, k] - position[i, j - 2, k]) / float(2 * rate)
            else:
                for k in range(position.shape[2]):
                    velocity[i, j, k] = (position[i, j + 1, k] - position[i, j - 1, k]) / float(2 * rate)
    return


def bce_loss(x, x_recon):
    x_recon = (x_recon + 1) / 2.0
    x = (x + 1) / 2.0
    return torch.nn.functional.binary_cross_entropy(x_recon.contiguous().view(-1, 50 * 4), x.view(-1, 50 * 4), reduction='sum') / x.size(0)


def kld_loss(z):
    mu = z[0]
    logvar = z[1]
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var