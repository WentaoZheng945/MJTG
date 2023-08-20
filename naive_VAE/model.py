import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import CUDA


class VAE(nn.Module):
    def __init__(self, z_dim, scene_len, scene_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim  # 隐向量维度 64
        self.seq_dim = scene_dim  # 输入时序数据的特征维度 6
        self.seq_len = scene_len  # 输入时序数据的长度 125

        self.input_size = 64 * 2  # GRU模块的特征维度的输入大小
        self.num_layers = 1  # GRU模块隐层数
        self.hidden_size_encoder = 128 * 2  # 编码器部分中GRU的隐层神经元个数
        self.hidden_size_decoder = 128 * 2  # 解码器部分中GRU的隐层神经元个数

        # 1. build encoder
        # -----------------------------------------------------------------
        # 全连接层接ReLU，实现输入数据到编码器GRU的连接，输出[batch_size, seq_len, 128]
        self.embedding_encoder = nn.Sequential(
            nn.Linear(self.seq_dim, self.input_size),
            nn.ReLU())  # 输出为[*, 125, 128]

        # 编码器的GRU模块，采样双向GRU，输出[batch_size, seq_len, 2 * hidden_size_encoder]
        # input_size指输入特征维度，hidden_size_encoder隐藏层特征维度，lstm隐层层数，batch_size是否为第一维度，bidirectional双向
        self.encoder_gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_encoder, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        # 编码器输出部分，实现从GRU输出到隐向量z的输出，输出[batch_size, 2 * z_dim]
        self.encoder = nn.Linear(2 * self.hidden_size_encoder, 2 * z_dim)  # 分别输出均值mu与方差的log值logvar

        # 2. build decoder
        # -----------------------------------------------------------------
        # 全连接层接ReLU，实现从隐向量z得到解码器GRU模块隐层神经元状态h
        self.z2hidden = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_size_decoder),
            nn.ReLU())

        # 全连接层接ReLU，实现从GRU隐层神经元状态得到解码器GRU模块的部分输入
        self.hidden2input = nn.Sequential(
            nn.Linear(self.hidden_size_decoder, int(self.input_size / 2)),
            nn.ReLU())

        # 解码器的GRU模块，采样单向GRU，输出[batch_size, seq_len, hidden_size_encoder]
        self.decoder_gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_decoder, num_layers=self.num_layers, batch_first=True)

        # 全连接层接ReLU，实现从上一帧时序数据切片x得到解码器GRU模块的部分输入
        self.embedding_decoder = nn.Sequential(
            nn.Linear(self.seq_dim, int(self.input_size / 2)),
            nn.ReLU())

        # 解码器输出部分，实现从GRU输出到时序张量单独帧切片的输出，输出[batch_size, 1, seq_dim]
        self.decoder = nn.Linear(self.hidden_size_decoder, self.seq_dim)

    def encoder_forward_(self, x):
        """编码器的前向更新函数，输入x [batch_size, seq_len, seq_dim]"""
        batch_size = x.size(0)  # [batch_size, 125, 6]
        x_embedding = self.embedding_encoder(x)  # [batch_size, 125, 128]

        # GRU
        hidden_encoder = self.init_hidden(batch_size, self.hidden_size_encoder)  # 初始化隐层神经元的输入为0
        output, _ = self.encoder_gru(x_embedding, hidden_encoder)  # [batch_size, 125, 512]
        # gru的输出为两项：output为隐层的输出，_为模型输出，output的最后一维其实就是_
        # output的维度为：[batch_size, seq_len, num_directions * hidden_size]
        # hn的维度为：[num_layers * num_directions, batch, hidden_size]
        # gru输入与输出关系维度参考：https://blog.csdn.net/jiuweideqixu/article/details/109492863
        # gru结构参数参考：https://blog.csdn.net/mch2869253130/article/details/103312364
        # 只取最后一帧送入最后的全连接层，将其压缩维度
        last_output = self.encoder(output[:, -1, :])  # [batch_size, 2 * self.z_dim]

        # output
        mu = last_output[:, :self.z_dim]  # [batch_size, 64]
        logvar = last_output[:, self.z_dim:]  # [batch_size, 64]
        z = self.reparametrize(mu, logvar)  # [batch_size, 64]
        return z, [mu, logvar]

    def decoder_forward(self, z):
        """解码器的前向更新函数，输入z [batch_size, z_dim], z_dim=64"""
        """解码共包含四个全连接层和一个单向GRU，四个全连接层分别为：z2hidden：把重采样得到的样本进行升维：->[batch_size, 1, 256]
        hidden2input:把隐层输出升维成降维成[batch_size, 1, 64], 作为gru输入的一部分
        embedding:通过这个模块得到gru每一帧的输入另一部分，初始帧是用0张量
        decoder:将得到的gru输出降维成我们需要的一帧轨迹储存，同时作为下一时刻embedding的输入"""
        """TODO:初始时，gru的隐层输入初始为z2hidden的输出，后续为上一次gru的隐层输出"""
        batch_size = z.size(0)

        # 初始化x输入，用于推导x0
        start_point = CUDA(Variable(torch.zeros(batch_size, 1, self.seq_dim)))  # 初始化第一帧为0 [batch_size, 1, 6]

        # z: [batch_size, 64] -> [batch_size, 1, 64] | output: [batch_size, 1, 256] -> [1, batch_size, 256]
        hidden_decoder = self.z2hidden(z.view(batch_size, 1, self.z_dim)).permute(1, 0, 2)  # 转换维度[1, batch_size, 256]
        hidden = self.hidden2input(hidden_decoder.permute(1, 0, 2))  # [batch_size, 1, 64]

        # start_point: [batch_size, 1, 4]
        current_stage = self.embedding_decoder(start_point)  # [batch_size, 1, 64]
        start_embedding = torch.cat((current_stage, hidden), dim=2)  # 在最后一个维度上拼接，[batch_size, 1, 128]

        x_recon = []
        for _ in range(self.seq_len):  # 逐帧解析
            # output: [batch_size, 1, 256] hidden_decoder: [1, batch_size, 256]
            output, hidden_decoder = self.decoder_gru(start_embedding, hidden_decoder)  # [batch_size, 1, 256]
            # output切片: [batch_size, 256] -> [batch_size, 6] -> x_cur: [batch_size, 1, 6]
            x_cur = self.decoder(output[:, -1, :]).view(batch_size, -1, self.seq_dim)  # 重建出的当前帧状态 [batch_size, 1, 6]
            x_recon.append(x_cur.view(batch_size, -1))  # x_recon [[batch_size, 6]]

            # 基于当前帧，得到下一帧GRU模块的输入
            current_stage = self.embedding_decoder(x_cur)  # x[t-1]: [batch_size, 1, 6]
            hidden = self.hidden2input(hidden_decoder.permute(1, 0, 2))  # h[t-1]: [batch_size, 1, 64]
            start_embedding = torch.cat((current_stage, hidden), dim=2)
        x_recon = torch.stack(x_recon, dim=0).permute(1, 0, 2)  # [batch_size, seq_len, seq_dim]  # torch.stack将列表中多个n-1维张量拼接成n维张量
        return x_recon

    def forward(self, x):
        """VAE前向更新函数"""
        z, z_bag = self.encoder_forward_(x)  # z_bag:[mu, logvar]   mu:[batch_size, 32] logvar:[batch_size, 32]
        x_recon = self.decoder_forward(z)
        return x_recon, z_bag

    def init_hidden(self, batch_size, hidden_size):
        """初始化编码器双向GRU的隐层神经元的输入h0状态"""
        # 在目前的pytorch中tensor已经具备了自动求导功能，不再需要使用Variable
        # GRU的隐层输入全置为0，其输入的维度为[num_layers * num_directions, batch, hidden_size]
        hidden_h = Variable(torch.zeros(self.num_layers * 2, batch_size, hidden_size))  # 所有的输入都要作为叶子节点
        if torch.cuda.is_available():
            hidden_h = hidden_h.cuda()  # 放到gpu上
        return hidden_h

    def reparametrize(self, mu, logvar):
        """重参数技巧"""
        """从一个非标准正太分布中采样的这个过程是随机的，梯度通过这个采样的过程传递（对均值和方差求导）
           这里使用重参数技巧，把从非标准正太分布中采样，转化为从标准正太分布中采样，再映射回到原分布中（采样点*标准差+均值）
           这样对标准差和均值就可以进行求导了
           PS：这里需要注意只需要对均值和标准差进行求导，因此这里的采样点的tensor对应的requires_grad为False
           理解可以参考https://zhuanlan.zhihu.com/p/542478018#的解释图"""
        std = logvar.div(2).exp()  # 求标准差
        eps = CUDA(Variable(std.data.new(std.size()).normal_()))  # normal是添加了实际噪声，默认是标准正太分布，等同于采样
        # 生成std尺寸一样的tensor并用标准正态分布填充
        return mu + std * eps  # 标准正态*标准差+均值，获得对应正态分布下的隐变量z的采样值
