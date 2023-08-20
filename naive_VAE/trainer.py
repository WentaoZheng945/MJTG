import os
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import mkdirs, CUDA, kaiming_init, kld_loss, mse_loss, set_seed, get_velocity
from model import VAE
from dataloader import Sequence_Dataset


class VAE_Trainer(object):
    def __init__(self, args):
        # 杂项定义
        self.continue_training = args.continue_training  # 是否是继续训练 default: False
        self.model_id = args.id  # 模型id default:0
        self.max_iteration = args.max_iteration  # default:20000
        self.print_iter = args.print_iter  # 多少次迭代输出一次loss default:1
        self.save_epoch = args.save_epoch  # default:10
        self.num_workers = args.num_workers  # default:0

        self.z_dim = args.z_dim  # 隐变量维度 64
        self.scene_len = args.scene_len  # default:125
        self.scene_dim = args.scene_dim  # 参数维度 default:6
        self.lr = args.lr  # 学习率  0.0001
        self.alpha = args.alpha
        self.beta = args.beta  # kld权重
        self.gama = args.gama  # 纵向速度权重 default:50
        self.weight = args.weight  # 位置上的权重 default[10, 1, 10, 1, 10, 1]

        self.training_data_path = args.training_data_path  # 训练集位置
        self.testing_data_path = args.testing_data_path  # 测试集位置

        self.latent_cons = args.latent_cons  # default:False
        self.batch_size = args.batch_size  # default:512
        self.test_batch_size = args.test_batch_size  # default:2000
        self.sample_batch_size = args.sample_batch_size  # default:1000

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 放到gpu上

        # checkpoint
        self.save_model_path = args.save_model_path
        mkdirs(self.save_model_path)  # 检查是否存在保存位置

        self.sample_path = args.sample_path
        mkdirs(self.sample_path)

        self.test_path = args.test_path
        mkdirs(self.test_path)

        self.loss_path = args.loss_path
        mkdirs(self.loss_path)

        # set seed
        set_seed(1)

        # model
        self.vae = CUDA(VAE(z_dim=self.z_dim, scene_len=self.scene_len, scene_dim=self.scene_dim))  # 把实例化模型放到gpu上
        self.optimizer = optim.Adam(params=self.vae.parameters(), lr=3 * self.lr)  # TODO 这里为啥学习率乘3
        if self.continue_training:
            self.load_model()  # 加载模型
            print('[*] Finish loading parameters from file')
        else:
            self.vae.apply(kaiming_init)  # 将kaiming_init方法递归应用于模型每一层
            print('[*] Finish building model')

        # data
        self.x = CUDA(Variable(torch.FloatTensor(self.batch_size, self.scene_len, self.scene_dim)))  # TODO

        # load data
        self.dataset = Sequence_Dataset(self.training_data_path)  # 对dataloader的输入数据进行魔法方法实现
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

    def train(self):
        loss_collector = []  # 损失列表
        pbar_epoch = tqdm(total=self.max_iteration, desc='[Epoch]')
        max_iteration = int(len(self.dataset) / self.batch_size)  # 计算迭代次数
        self.vae.train()  # TODO: 把模型调整训练模式
        for epoch in range(self.max_iteration):
            pbar_iteration = tqdm(total=max_iteration, desc='[Iteration]')
            iteration = 0
            for x in self.dataloader:
                self.batch_size = x.size(0)
                self.x.resize_(self.batch_size, self.scene_len, self.scene_dim)

                pbar_iteration.update(1)  # 进度条加1
                iteration += 1
                self.x.copy_(x)  # 保留self.x的shape，把x内容复制到self.x中，这里仅进行数据的复制，其余所有属性都与self.x一致，我觉得这里是为了解决，每个epoch最后一个iter数据量不足的问题，

                x_recon, z_bag = self.vae(self.x)  # 将编码器的隐变量空间均值和方差提取出来计算kld

                # kl divergence
                kld = kld_loss(z_bag)

                # reconstruction loss
                recon = [0] * self.x.shape[-1]
                for i in range(self.x.shape[-1]):
                    recon[i] = mse_loss(self.x[:, :, i], x_recon[:, :, i]) * self.weight[i]  # 对每个特征维度分别计算MSE，加权
                recon_loss = sum(recon)  # MSE求和

                # velocity loss
                velocity = torch.diff(self.x, dim=1) / 0.04
                velocity_recon = torch.diff(x_recon, dim=1) / 0.04
                # 这里只计算纵向速度误差
                velocity_loss = mse_loss(velocity[:, :, 0], velocity_recon[:, :, 0]) + mse_loss(velocity[:, :, 2], velocity_recon[:, :, 2]) + mse_loss(velocity[:, :, 4], velocity_recon[:, :, 4])

                total_loss = recon_loss * self.alpha + kld * self.beta + velocity_loss * self.gama

                # 清空梯度->反向传播->更新
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if iteration % self.print_iter == 0:
                    pbar_iteration.write('[%d/%d] kld: %.6f, position: %.6f, velocity: %.6f, total_loss: %.6f' %
                                         (iteration, max_iteration, kld.detach().cpu().numpy(), recon_loss.detach().cpu().numpy(), velocity_loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy()))
                    loss_collector.append([epoch, iteration, kld.detach().cpu().numpy(), recon_loss.detach().cpu().numpy(), velocity_loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy()])

            # save model
            if epoch % self.save_epoch == 0:
                self.save_model()
                pbar_iteration.write('[*] Save one model')
                np.save(self.loss_path + '/loss' + str(self.model_id) + '.npy', np.array(loss_collector))

            pbar_iteration.close()
            pbar_epoch.update(1)

        pbar_epoch.write("[*] Training stage finishes")
        pbar_epoch.close()
        return

    def test(self):
        """由测试集验证重建效果"""
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        dataset = Sequence_Dataset(self.testing_data_path)
        self.test_batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
        test_dataset = np.zeros((len(dataset), self.scene_len, self.scene_dim))
        pbar_iteration = tqdm(total=int(len(dataset) / self.test_batch_size), desc='[Iteration]')
        count = 0
        self.vae.eval()  # 把模型调整预测模式
        with torch.no_grad():  # 关闭梯度计算
            for x in dataloader:
                self.test_batch_size = x.size(0)
                self.x.resize_(self.test_batch_size, self.scene_len, self.scene_dim)
                self.x.copy_(x)

                z, _ = self.vae.encoder_forward_(self.x)
                x_recon = self.vae.decoder_forward(z)

                test_dataset[(count * self.test_batch_size):((count + 1) * self.test_batch_size)] = x_recon.detach().cpu().numpy()
                count += 1
                pbar_iteration.update(1)

        np.save(os.path.join(self.test_path, 'data_for_vaild.npy'), test_dataset)
        pbar_iteration.write('[*] Finish generating test dataset')
        pbar_iteration.close()

    def sample(self):
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        sample_dataset = np.zeros((self.sample_batch_size, self.scene_len, self.scene_dim))
        self.vae.eval()
        with torch.no_grad():
            if not self.latent_cons:  # 传统隐空间采样，不做约束
                sample_z = CUDA(torch.randn(self.sample_batch_size, self.z_dim))
                sample_x = self.vae.decoder_forward(sample_z)
                sample_dataset = sample_x.detach().cpu().numpy()
                np.save(os.path.join(self.sample_path, 'data_without_latent_cons.npy'), sample_dataset)
        print('[*] Finish generating sample dataset')

    def save_model(self):
        states = {'vae_states': self.vae.state_dict(), 'optim_states': self.optimizer.state_dict()}

        filepath = os.path.join(self.save_model_path, 'model.' + str(self.model_id) + '.torch')
        with open(filepath, 'wb+') as f:  # 二进制写入，以存在则覆盖
            torch.save(states, f)

    """
    模型的保存有两种方式：
    一种是保存模型的state_dict(),只保存模型的参数。这种方式在加载时需要先创建模型的实例model，之后通过torch.load()将模型参数加载进来，
    得到dict，再通过model.load_state_dict(dict)将参数更新。
    第二种方式是将整个模型都保存下来，加载时通过torch.load()将模型整个加载进来，返回加载好的模型
    """

    def load_model(self):
        """
        若模型为继续训练模型，则加载模型的参数与优化器的参数
        """
        filepath = os.path.join(self.save_model_path, 'model.' + str(self.model_id) + '.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                # torch.load加载模型时，会把上次训练时模型参数的位置也一起加载下来
                # （比如：上一次在cuda:0上，这次加载后打印next(model.parameters()).device还在原设备上
                # map_location用于改变加载的位置，这个过程叫做重定向
                checkpoint = torch.load(f, map_location=self.device)  # 加载模型参数到GPU上

            self.vae.load_state_dict(checkpoint['vae_states'])
            self.optimizer.load_state_dict(checkpoint['optim_states'])
