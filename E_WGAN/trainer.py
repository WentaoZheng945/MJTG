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
        self.continue_training = args.continue_training
        self.model_id = args.id
        self.batch_size = args.batch_size
        self.sample_batch_size = args.sample_batch_size
        self.max_epoch = args.max_epoch
        self.print_iter = args.print_iter
        self.save_epoch = args.save_epoch
        self.num_workers = args.num_workers

        self.z_dim = args.z_dim
        self.scene_len = args.scene_len
        self.scene_dimension = args.scene_dimension
        self.lr = args.lr
        self.num_layer = args.num_layer
        self.weight_dp = args.distance_penalty
        self.weight_cliping_limit = args.weight_cliping_limit

        self.vae_model_path = args.vae_model_path
        self.vae_model_id = args.vae_model_id

        self.training_data_path = args.training_data_path
        self.testing_data_path = args.testing_data_path

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # checkpoint
        self.save_model_path = args.save_model_path
        mkdirs(self.save_model_path)

        self.sample_path = args.sample_path
        mkdirs(self.sample_path)

        self.loss_path = args.loss_path
        mkdirs(self.loss_path)

        # load vae model
        self.vae_model = model.VAE(z_dim=self.z_dim, scene_len=self.scene_len, scene_dim=self.scene_dimension)
        filepath = os.path.join(self.vae_model_path, 'model.' + str(self.vae_model_id) + '.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.vae_model.load_state_dict(checkpoint['vae_states'])
            print('[*] Finish loading vae model from file')
        else:
            print('[*] Finish building vae model')
        self.vae_model.to(self.device)

        # bulid gan model
        self.actor = Actor(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Generator
        self.actor_optim = optim.RMSprop(params=self.actor.parameters(), lr=self.lr)
        self.critic = Critic(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Discrimintor
        self.critic_optim = optim.RMSprop(params=self.critic.parameters(), lr=self.lr)
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
        self.vae_model.eval()
        self.actor.train()
        self.critic.train()
        if self.continue_training:
            loss_collector = np.load(self.loss_path + './loss' + str(self.model_id) + '.npy', allow_pickle=True).tolist()
            base_epoch = loss_collector[-1][0]
        else:
            loss_collector = []
            base_epoch = 0
        iteration = 0
        for epoch in range(self.max_epoch):
            # pbar_iteration = tqdm(total=max_iteration, desc='[Iteration]')
            d_loss = 0
            g_loss = 0
            Wasserstein_D = 0
            for batch_idx, data in enumerate(self.trainDataLoader):
                iteration += 1

                # check input data
                self.batch_size = data.size(0)
                data = data.to(self.device)

                # set data label
                real_data_label = torch.ones(self.batch_size, 1)
                real_data_label = real_data_label.to(self.device)
                fake_data_label = torch.zeros(self.batch_size, 1)
                fake_data_label = fake_data_label.to(self.device)

                # set representation z
                fake_z = Variable(torch.randn(self.batch_size, self.z_dim))
                fake_z = fake_z.to(self.device)
                with torch.no_grad():
                    _, z_bag = self.vae_model.encoder_forward_(data.to(torch.float32))
                real_z = self.reparametrize(z_bag[0], z_bag[1])
                real_z = real_z.to(self.device)

                # train D
                if iteration:
                    for p in self.critic.parameters():
                        p.requires_grad = True

                    fake_z_g = self.actor(fake_z)
                    input_fake_data = fake_z_g
                    input_real_date = real_z
                    d_loss_real = self.critic(input_real_date)
                    d_loss_real = d_loss_real.mean(0).view(1)

                    d_loss_fake = self.critic(input_fake_data)
                    d_loss_fake = d_loss_fake.mean(0).view(1)

                    d_loss = d_loss_fake - d_loss_real
                    Wasserstein_D = d_loss_real - d_loss_fake
                    self.critic.zero_grad()
                    d_loss.backward()
                    self.critic_optim.step()

                    # Clamp paramters to a range [-c, c]
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                    '''
                    if (iteration + 1) % 10 == 0:
                        self.d_critic_histogram(iteration)
                    '''
                    # total_real_loss += critic_loss.item()

                # train G
                if iteration % 20 == 0:
                    for p in self.critic.parameters():
                        p.requires_grad = False

                    fake_z_g = self.actor(fake_z)
                    g_loss = self.critic(fake_z_g)
                    g_loss = g_loss.mean(0).view(1) * -1

                    self.actor.zero_grad()
                    g_loss.backward()
                    self.actor_optim.step()
                    # self.d_actor_histogram(iteration)

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
        self.vae_model.eval()
        self.actor.eval()
        with torch.no_grad():
            normal_z = Variable(torch.randn(self.sample_batch_size, self.z_dim))
            normal_z = normal_z.to(self.device)

            processed_z = self.actor(normal_z)

            processed_x = self.vae_model.decoder_forward(processed_z)
            processed_dataset = processed_x.detach().cpu().numpy()

            original_x = self.vae_model.decoder_forward(normal_z)
            original_dataset = original_x.detach().cpu().numpy()
            np.save(os.path.join(self.sample_path, 'data_without_latent_cons.npy'), original_dataset)
            np.save(os.path.join(self.sample_path, 'data_with_latent_cons.npy'), processed_dataset)
        print('[*] Finish generating sample dataset')

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def _summary_wrtie(self, distance_penalty, loss, real_loss, epoch):
        self.tensorboad_writer.add_scalar('data/loss', loss, epoch)  # need to modify . We use four loss value .
        self.tensorboad_writer.add_scalar('data/distance_penalty', distance_penalty, epoch)  # need to modify . We use four loss value
        self.tensorboad_writer.add_scalar('data/discriminator_loss', real_loss, epoch)

    def d_critic_histogram(self, iteration):
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
