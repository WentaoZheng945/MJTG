import os
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib as mpl
mpl.use('Agg')


def set_seed(seed):
    # 为cpu和gpu设置随机种子
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mse_loss(x, x_recon):
    """
    计算解码器对隐变量z解码后生成的重建场景与输入场景之间的均方误差MSE
    值得注意的是，此处的MSE中“均”字的落脚点是在各场景的均值，也即先计算每个场景对之间的和方差SSE，再对该批次下所有场景对进行平均，得到MSE

    输出当前批次下所有场景与其重建场景之间的SSE均值，即均方误差MSE
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
    """
    计算编码器生成隐变量z的分布与标准正态分布之间的KL散度
    kl[q||N(0,1)] = logvar + [(var + mu^2) / 2] - 1 / 2 = -0.5 * [1 + logvar - var - mu^2]
    输出当前批次下所有样本对应隐变量z分布的KLD
    """
    mu = z[0]
    logvar = z[1]
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def kaiming_init(m):
    """
    深度学习网络参数初始化设置
    """
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)  # Xavier方法初始化m层权重
        if m.bias is not None:
            m.bias.data.fill_(0)  # 初始化bias为0
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)  # Kaiming方法初始化m层权重
        if m.bias is not None:
            m.bias.data.fill_(0)  # 初始化bias为0
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var