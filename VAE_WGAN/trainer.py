import os
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from tensorboardX import SummaryWriter

from model import Actor, Critic
from utils import mkdirs
from dataloader import Sequence_Dataset

import sys
sys.path.append('..')
from naive_VAE import model


class AC_Trainer(object):
    def __init__(self, args):
        # 杂项定义
        self.continue_training = args.continue_training  # 是否继续训练
        self.model_id = args.id  # 模型存储id
        self.batch_size = args.batch_size  # 训练批大小
        self.sample_batch_size = args.sample_batch_size  # 采样批大小
        self.max_epoch = args.max_epoch  # 最大epoch
        self.print_iter = args.print_iter
        self.save_epoch = args.save_epoch
        self.num_workers = args.num_workers
        # self.tensorboad_writer = SummaryWriter()

        self.z_dim = args.z_dim  # 隐变量维度 64
        self.scene_len = args.scene_len  # 长度 125
        self.scene_dimension = args.scene_dimension  # 场景维度 6
        self.lr = args.lr  # 学习率
        self.num_layer = args.num_layer  # GAN的G和D层数 default:2
        self.weight_dp = args.distance_penalty  # 正则项权重
        self.weight_cliping_limit = args.weight_cliping_limit  # 参数截断阈值

        self.vae_model_path = args.vae_model_path  # vae模型的存储路径
        self.vae_model_id = args.vae_model_id  # vae模型的id

        self.training_data_path = args.training_data_path  # 训练集的数据位置
        self.testing_data_path = args.testing_data_path  # 测试集的数据位置

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备编号

        # checkpoint
        self.save_model_path = args.save_model_path  # GAN模型保存位置
        mkdirs(self.save_model_path)

        self.sample_path = args.sample_path
        mkdirs(self.sample_path)

        self.loss_path = args.loss_path
        mkdirs(self.loss_path)

        self.test_path = args.test_path  # 重建数据的保存位置
        mkdirs(self.test_path)

        # load vae model
        self.vae_model = model.VAE(z_dim=self.z_dim, scene_len=self.scene_len, scene_dim=self.scene_dimension)  # 实例化一个VAE模型
        filepath = os.path.join(self.vae_model_path, 'model.' + str(self.vae_model_id) + '.torch')  # 模型参数位置
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.vae_model.load_state_dict(checkpoint['vae_states'])  # 加载VAE模型参数
            print('[*] Finish loading vae model from file')
        else:
            print('[*] Finish building vae model')
        self.vae_model.to(self.device)  # 模型放到GPU上

        # bulid gan model
        self.actor = Actor(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Generator
        self.actor_optim = optim.RMSprop(params=self.actor.parameters(), lr=self.lr)  # G的学习率为0.00005
        self.critic = Critic(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Discrimintor
        self.critic_optim = optim.RMSprop(params=self.critic.parameters(), lr=self.lr)  # D的学习率为0.00005
        self.actor.to(self.device)
        self.critic.to(self.device)

        if self.continue_training:
            self.load_model(self.actor, self.actor_optim, self.save_model_path, self.model_id, 'actor')
            self.load_model(self.critic, self.critic_optim, self.save_model_path, self.model_id, 'critic')
            print('[*] Finish loading gan model from file')
        else:
            print('[*] Finish build gan model')

        # load data
        self.trainDataset = Sequence_Dataset(self.training_data_path)
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

    def train(self):
        pbar_epoch = tqdm(total=self.max_epoch, desc='[Epoch]')
        # max_iteration = int(len(self.trainDataset) / self.batch_size)
        # 设定各模型模式，vae不参与训练，actor与critic参与训练
        self.vae_model.eval()
        self.actor.train()
        self.critic.train()
        if self.continue_training:
            loss_collector = np.load(self.loss_path + './loss' + str(self.model_id) + '.npy', allow_pickle=True).tolist()
            base_epoch = loss_collector[-1][0]
        else:
            loss_collector = []
            base_epoch = 0
        iteration = 0  # 记录整个训练过程的所有iteration数
        for epoch in range(self.max_epoch):
            # pbar_iteration = tqdm(total=max_iteration, desc='[Iteration]')
            d_loss = 0  # 初始化损失，后续计算
            g_loss = 0
            Wasserstein_D = 0
            for batch_idx, data in enumerate(self.trainDataLoader):
                iteration += 1
                # pbar_iteration.update(1)

                # check input data
                self.batch_size = data.size(0)  # 每次都需要确认batch_size，因为存在batch_size无法除尽的情况
                data = data.to(self.device)  # 数据也要放在gpu上

                # set data label
                real_data_label = torch.ones(self.batch_size, 1)  # 正类标签
                real_data_label = real_data_label.to(self.device)
                fake_data_label = torch.zeros(self.batch_size, 1)  # 反类标签
                fake_data_label = fake_data_label.to(self.device)

                # set representation z
                fake_z = Variable(torch.randn(self.batch_size, self.z_dim))  # 负类，服从先验分布p(z)的隐向量（由标准正态分布随机生成）
                fake_z = fake_z.to(self.device)
                with torch.no_grad():
                    _, z_bag = self.vae_model.encoder_forward_(data.to(torch.float32))  # 正类，由输入经编码器到隐变量
                real_z = self.reparametrize(z_bag[0], z_bag[1])  # 正类，直接由VAE编码器得到的隐向量，服从z~q(z|xn)
                real_z = real_z.to(self.device)

                # train D
                '''G与D联合训练的思路：
                G与D按照1：20的步长进行训练，即D每训练20个iteration，G便训练1个iteration（先训练20次D，再训练一次G）
                对于D的训练，由于来自先验分布p(z)的样本比来自于经过G处理偏移后的样本G(p(z))更易区分，故以比G(p(z))低10倍的速率从p(z)采样来训练D
                '''
                if iteration:
                    # 激活D
                    for p in self.critic.parameters():
                        p.requires_grad = True

                    # 从G(p(z))中采样作为输入
                    fake_z_g = self.actor(fake_z)  # 生成器生成的假数据
                    # WGAN输入
                    input_fake_data = fake_z_g  # 假数据
                    input_real_date = real_z  # 真数据
                    # 计算Critic损失
                    d_loss_real = self.critic(input_real_date)  # 真数据损失  (越大越好)
                    d_loss_real = d_loss_real.mean(0).view(1)  # output为一个tensor里面只有一个值(0是指第一个维度，view(1)是因为里面只有一个值)

                    d_loss_fake = self.critic(input_fake_data)  # 假数据损失  (越小越好)
                    d_loss_fake = d_loss_fake.mean(0).view(1)

                    d_loss = d_loss_fake - d_loss_real
                    Wasserstein_D = d_loss_real - d_loss_fake  # W距离为-loss
                    self.critic.zero_grad()  # 清零
                    d_loss.backward()  # 回传
                    self.critic_optim.step()  # 更新

                    # Clamp paramters to a range [-c, c]  # 更新后，实现网络参数截断
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                    '''
                    if (iteration + 1) % 10 == 0:
                        self.d_critic_histogram(iteration)
                    '''
                    # total_real_loss += critic_loss.item()  # 由于仅记录模型损失，并不参与梯度更新，故需要.item()去除Varible中涉及梯度的部分信息，否则会使计算图无限增大

                # train G
                if iteration % 20 == 0:
                    # 冻结D
                    for p in self.critic.parameters():
                        p.requires_grad = False

                    # 生成器对隐向量进行偏移
                    fake_z_g = self.actor(fake_z)  # 先验分布采样得到的隐向量经过G偏移
                    g_loss = self.critic(fake_z_g)
                    g_loss = g_loss.mean(0).view(1) * -1

                    self.actor.zero_grad()
                    g_loss.backward()
                    self.actor_optim.step()
                    # self.d_actor_histogram(iteration)

                # 每隔固定iteration，记录一次各类损失
                if iteration % self.print_iter == 0:
                    pbar_epoch.write('[%d] Wasserstein_D: %.6f, D_loss: %.6f, G_loss: %.6f' %
                                     (epoch, Wasserstein_D.detach().cpu().numpy(), d_loss.detach().cpu().numpy(), g_loss.detach().cpu().numpy()))
                    loss_collector.append([epoch + base_epoch, iteration, Wasserstein_D.detach().cpu().numpy()[0], d_loss.detach().cpu().numpy()[0], g_loss.detach().cpu().numpy()[0]])

            # save model
            if epoch % self.save_epoch == 0:
                self.save_model()
                pbar_epoch.write('[*] Saved one model')
                np.save(self.loss_path + './loss' + str(self.model_id) + '.npy', np.array(loss_collector))

            # pbar_iteration.close()
            pbar_epoch.update(1)
        pbar_epoch.write('[*] Training stage finished')
        pbar_epoch.close()
        return

    def sample(self):
        # load model
        self.load_model(self.actor, self.actor_optim, self.save_model_path, self.model_id, 'actor')
        print('[*] Finish loading Actor')

        processed_dataset = np.zeros((self.sample_batch_size, self.scene_len, self.z_dim))
        original_dataset = np.zeros((self.sample_batch_size, self.scene_len, self.z_dim))
        self.vae_model.eval()  # 切换到评估模式
        self.actor.eval()
        with torch.no_grad():
            normal_z = Variable(torch.randn(self.sample_batch_size, self.z_dim))  # 基于先验分布采样得到的隐向量
            normal_z = normal_z.to(self.device)

            processed_z = self.actor(normal_z)  # 经过GAN修正的隐向量

            processed_x = self.vae_model.decoder_forward(processed_z)  # 隐向量由Decoder解码成场景
            processed_dataset = processed_x.detach().cpu().numpy()

            original_x = self.vae_model.decoder_forward(normal_z)
            original_dataset = original_x.detach().cpu().numpy()
            np.save(os.path.join(self.sample_path, 'data_without_latent_cons.npy'), original_dataset)
            np.save(os.path.join(self.sample_path, 'data_with_latent_cons.npy'), processed_dataset)
        print('[*] Finish generating sample dataset')

    def test(self):
        """由测试集验证重建效果"""

        dataset = Sequence_Dataset(self.testing_data_path)
        self.test_batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
        test_dataset = np.zeros((len(dataset), self.scene_len, self.scene_dimension))
        pbar_iteration = tqdm(total=int(len(dataset) / self.test_batch_size), desc='[Iteration]')

        count = 0
        self.vae_model.eval()
        with torch.no_grad():
            for x in dataloader:
                self.test_batch_size = x.size(0)
                x.resize_(self.test_batch_size, self.scene_len, self.scene_dimension)
                x = x.float()

                z, _ = self.vae_model.encoder_forward_(x)
                x_recon = self.vae_model.decoder_forward(z)

                test_dataset[(count * self.test_batch_size):((count+1)*self.test_batch_size)] = x_recon.detach().cpu().numpy()
                count += 1
                pbar_iteration.update(1)

        np.save(os.path.join(self.test_path, 'data_for_valid.npy'), test_dataset)
        pbar_iteration.write('[*] Finish generating test dataset')
        pbar_iteration.close()
        return


    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())  # 生成std尺寸一样的tensor并用标准正态分布填充
        return mu + std * eps  # 标准正态*标准差+均值，获得对应正态分布下的隐变量z的采样值

    def _summary_wrtie(self, distance_penalty, loss, real_loss, epoch):
        '''记录训练过程中的相关数据，形成对应的日志文件
        add_scalar(): 向日志文件中写入标量数据，用于绘制图标，三个参数分别对应图表标题、图表中的y轴数据、图表中的x轴数据
        '''
        self.tensorboad_writer.add_scalar('data/loss', loss, epoch)  # need to modify . We use four loss value .
        self.tensorboad_writer.add_scalar('data/distance_penalty', distance_penalty, epoch)  # need to modify . We use four loss value
        self.tensorboad_writer.add_scalar('data/discriminator_loss', real_loss, epoch)

    def d_critic_histogram(self, iteration):
        '''记录模型各参数（包含梯度）的分布情况，用于监视模型训练情况'''
        for name, param in self.critic.named_parameters():  # actor
            self.tensorboad_writer.add_histogram('real_critic/' + name, param.clone().cpu().data.numpy(), iteration, bins='sturges')
            self.tensorboad_writer.add_histogram('real_critic/' + name + '/grad', param.grad.clone().cpu().data.numpy(), iteration, bins='sturges')

    def d_actor_histogram(self, iteration):
        for name, param in self.actor.named_parameters():  # actor
            self.tensorboad_writer.add_histogram('actor/' + name, param.clone().cpu().data.numpy(), iteration, bins='sturges')
            self.tensorboad_writer.add_histogram('actor/' + name + ' /grad', param.grad.clone().cpu().data.numpy(), iteration, bins='sturges')

    def save_model(self):
        # actor
        states = {'actor_states': self.actor.state_dict(), 'actor_optim_states': self.actor_optim.state_dict()}
        filepath = os.path.join(self.save_model_path, 'actor.' + str(self.model_id) + '.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        # critic
        states = {'critic_states': self.critic.state_dict(), 'critic_optim_states': self.critic_optim.state_dict()}
        filepath = os.path.join(self.save_model_path, 'critic.' + str(self.model_id) + '.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        return

    def load_model(self, model, optim, path, id, type):
        filepath = os.path.join(path, type + '.' + str(id) + '.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)

            model.load_state_dict(checkpoint[(type + '_states')])
            optim.load_state_dict(checkpoint[(type + '_optim_states')])
        return
