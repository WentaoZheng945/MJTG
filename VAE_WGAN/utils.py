'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-02-20 13:54:50
@LastEditTime: 2020-02-20 14:01:42
@Description:
'''

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mse_loss(x, x_recon):
    '''
    计算解码器对隐变量z解码后生成的重建场景与输入场景之间的均方误差MSE
    值得注意的是，此处的MSE中“均”字的落脚点是在各场景的均值，也即先计算每个场景对之间的和方差SSE，再对该批次下所有场景对进行平均，得到MSE

    输出当前批次下所有场景与其重建场景之间的SSE均值，即均方误差MSE
    '''
    return torch.nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)


def bce_loss(x, x_recon):
    x_recon = (x_recon + 1) / 2.0
    x = (x + 1) / 2.0
    return torch.nn.functional.binary_cross_entropy(x_recon.contiguous().view(-1, 50 * 4), x.view(-1, 50 * 4), reduction='sum') / x.size(0)


def kld_loss(z):
    '''
    计算编码器生成隐变量z的分布与标准正态分布之间的KL散度
    kl[q||N(0,1)] = logvar + [(var + mu^2) / 2] - 1 / 2 = -0.5 * [1 + logvar - var - mu^2]

    输出当前批次下所有样本对应隐变量z分布的KLD
    '''
    mu = z[0]
    logvar = z[1]
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.0099999)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.0099999)


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
