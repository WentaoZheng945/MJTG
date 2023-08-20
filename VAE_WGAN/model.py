import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """GAN中的Generator部分
    作用：对VAE生成的隐向量z做偏移，使其从服从p(z)变为服从z~q(z|xn)
    结构：连续三层跟着ReLU的线性层，输出均为2 * z.size(0)，前半部分为偏移δz，后半部分输入sigmoid层用作gate
    输出：z' = (1 - gate) * z + gate * δz  (这里我理解是修正后的偏移与原始z做加权)
    优势：Generator只需学习如何从预设分布p(z)做偏移至q(z|xn)，而不用直接学习如何从噪声数据中生成符合q(z|xn)的向量，有利于训练
    """
    def __init__(self, d_z, d_model, layer_num=3):
        super(Actor, self).__init__()
        self.d_z = d_z  # 输入的隐向量z (num, z_dim)  default:64
        self.d_model = d_model  # G的隐层神经元个数 default: 4*64
        self.layer_num = layer_num  # G的网络层数 default: 2
        layer_list = []
        for i in range(layer_num):
            '''定义G的网络结构，三层线性层（ReLu激活）'''
            if i == 0:
                input_dim = d_z  # 输入为[batch_size, 64]
                layer_list.append(nn.Linear(input_dim, self.d_model))  # 第一层输入为z_dim, 输出为d_model   第一层输出为[batch_size, 256]
            else:
                layer_list.append(nn.Linear(self.d_model, self.d_model))  # 第二个全连接层输出为[batch_size, 256]
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))  # 每层的激活函数均为ReLu
        layer_list.append(nn.Linear(self.d_model, self.d_z * 2))  # 最后一层输出为[batch_size, 128]

        self.fw_layer = nn.Sequential(*layer_list)  # 将定义的网络结构，顺序传入nn.Sequential容器中，封装成一个完整的网络模块
        self.gate = nn.Sigmoid()  # 最后针对fw_layer的一半输出，通过sigmoid层计算gate

    def forward(self, x):
        original_x = x
        out = self.fw_layer(x)  # 输出张量(num, 2 * z_dim)
        input_gate, dz = out.chunk(2, dim=-1)  # 将输出张量平均分块2 * (num, z_dim)，分别对应gate层的输入，与z向量的偏移
        gate_value = self.gate(input_gate)  # 通过sigmoid计算最终的gate权重
        new_z = (1 - gate_value) * original_x + gate_value * dz  # 得到最终偏移后的隐向量z' shape:[batch_size, 64]
        return new_z


class Critic(nn.Module):
    '''GAN中的Discriminator部分
    作用：分辨正类与负类——正类：符合z~q(z|xn)，负类：decoder输出的隐向量z~p(z)与生成器G输出的隐向量z~G(z)
    结构：连续四层采用ReLu激活的线性层，与最后一层采用sigmoid激活的线性层（WGAN则去除最后一层sigmoid函数），输出各样本是正类的概率D(z)
    '''
    def __init__(self, d_z, d_model, layer_num=3, num_output=1):
        super(Critic, self).__init__()
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
        return out  # WGAN 去除最后的sigmoid层
