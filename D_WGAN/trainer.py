import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mkdirs
import sys
sys.path.append('..')
from naive_VAE import model as vae_model
from E_WGAN import model as gan_model
from dataloader import Sequence_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from model import Actor_2, Critic_2
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable


class AC_Trainer_2(object):
    def __init__(self, args):
        self.continue_training =  args.continue_training
        self.model_id = args.id
        self.batch_size_danger = args.batch_size_danger
        self.batch_size_safe = args.batch_size_safe
        self.sample_batch_size = args.sample_batch_size
        self.max_epoch = args.max_epoch  # 最大epoch
        self.print_iter = args.print_iter
        self.save_epoch = args.save_epoch
        self.num_workers = args.num_workers
        # self.tensorboad_writer = SummaryWriter()

        self.z_dim = args.z_dim
        self.scene_len = args.scene_len
        self.scene_dimension = args.scene_dimension
        self.lr = args.lr
        self.num_layer = args.num_layer
        self.weight_dp = args.distance_penalty
        self.weight_cliping_limit = args.weight_cliping_limit

        self.vae_model_path = args.vae_model_path
        self.vae_model_id = args.vae_model_id
        self.first_gan_model_path = args.first_gan_model_path
        self.first_gan_model_id = args.first_gan_model_id

        self.training_data_danger_path = args.training_data_danger_path
        self.training_data_safe_path = args.training_data_safe_path
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

        self.test_path = args.test_path
        mkdirs(self.test_path)

        # load vae model
        self.vae_model = vae_model.VAE(z_dim=self.z_dim, scene_len=self.scene_len,
                                   scene_dim=self.scene_dimension)
        filepath = os.path.join(self.vae_model_path, 'model.' + str(self.vae_model_id) + '.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.vae_model.load_state_dict(checkpoint['vae_states'])
            print('[*] Finish loading vae model from file')
        else:
            print('[*] Finish building vae model')
        self.vae_model.to(self.device)

        # load first gan model
        self.first_gan_model_actor = gan_model.Actor(self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)
        filepath_gan = os.path.join(self.first_gan_model_path, 'actor.' + str(self.first_gan_model_id) + '.torch')
        if os.path.isfile(filepath_gan):
            with open(filepath_gan, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)
            self.first_gan_model_actor.load_state_dict(checkpoint['actor_states'])
            print('[*] Finish loading gan model from file')
        else:
            print('[*] Finish building gan model')
        self.first_gan_model_actor.to(self.device)

        # bulid second gan model
        self.actor = Actor_2(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Generator
        self.actor_optim = optim.RMSprop(params=self.actor.parameters(), lr=self.lr)
        self.critic = Critic_2(d_z=self.z_dim, d_model=4 * self.z_dim, layer_num=self.num_layer)  # Discrimintor
        self.critic_optim = optim.RMSprop(params=self.critic.parameters(), lr=self.lr)
        self.actor.to(self.device)
        self.critic.to(self.device)

        # check whether continue training or not
        if self.continue_training:
            self.load_model(self.actor, self.actor_optim, self.save_model_path, self.model_id, 'actor')
            self.load_model(self.critic, self.critic_optim, self.save_model_path, self.model_id, 'critic')
            print('[*] Finish loading gan model from file')
        else:
            print('[*] Finish build gan model')

        # load data
        self.trainDangerDataset = Sequence_Dataset(self.training_data_danger_path)
        self.trainDangerDataLoader = DataLoader(self.trainDangerDataset, batch_size=self.batch_size_danger, shuffle=True, num_workers=self.num_workers)
        self.trainSafeDataset = Sequence_Dataset(self.training_data_safe_path)
        self.trainSafeDataloader = DataLoader(self.trainSafeDataset, batch_size=self.batch_size_safe, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

    def train(self):
        pbar_epoch = tqdm(total=self.max_epoch, desc='[Epoch]')
        self.vae_model.eval()
        self.first_gan_model_actor.eval()
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
            d_loss = 0
            g_loss = 0
            Wasserstein_D = 0
            for data_danger, data_safe in zip(self.trainDangerDataLoader, self.trainSafeDataloader):
                iteration += 1

                # check input data
                self.danger_batch_size = data_danger.size(0)
                # data_danger = data_danger.to(self.device)

                self.safe_batch_size = data_safe.size(0)
                # data_safe = data_safe.to(self.device)

                combined_batch = torch.cat([data_danger, data_safe], dim=0)
                combined_batch = combined_batch.to(self.device)

                # set representation z
                with torch.no_grad():
                    _, z_bag = self.vae_model.encoder_forward_(combined_batch.to(torch.float32))
                data_z = self.reparametrize(z_bag[0], z_bag[1])
                data_z = data_z.to(self.device)
                with torch.no_grad():
                    data_z_g = self.first_gan_model_actor(data_z)


                danger_z = data_z_g[0:self.danger_batch_size, :]
                safe_z = data_z_g[self.danger_batch_size:, :]

                # train D
                if iteration:
                    for p in self.critic.parameters():
                        p.requires_grad = True

                    safe_z_g = self.actor(safe_z)
                    input_fake_data = safe_z_g
                    input_real_data = danger_z

                    d_loss_real = self.critic(input_real_data)
                    d_loss_real = d_loss_real.mean(0).view(1)

                    d_loss_fake = self.critic(input_fake_data)
                    d_loss_fake = d_loss_fake.mean(0).view(1)

                    d_loss = d_loss_fake - d_loss_real
                    Wasserstein_D = d_loss_real - d_loss_fake
                    self.critic.zero_grad()
                    d_loss.backward()
                    self.critic_optim.step()

                    for p in self.critic.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                # train G
                if iteration % 20 == 0:
                    for p in self.critic.parameters():
                        p.requires_grad = False

                    fake_z_g = self.actor(safe_z)
                    g_loss = self.critic(fake_z_g)
                    g_loss = g_loss.mean(0).view(1) * -1

                    self.actor.zero_grad()
                    g_loss.backward()
                    self.actor_optim.step()

                if iteration % self.print_iter == 0:
                    pbar_epoch.write('[%d] Wasserstein_D: %.6f, D_loss: %.6f, G_loss: %.6f' %
                                     (epoch, Wasserstein_D.detach().cpu().numpy(), d_loss.detach().cpu().numpy(),
                                      g_loss.detach().cpu().numpy()))
                    loss_collector.append([epoch + base_epoch, iteration, Wasserstein_D.detach().cpu().numpy()[0],
                                           d_loss.detach().cpu().numpy()[0], g_loss.detach().cpu().numpy()[0]])

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
        self.first_gan_model_actor.eval()
        self.actor.eval()
        with torch.no_grad():
            normal_z = Variable(torch.randn(self.sample_batch_size, self.z_dim))
            normal_z = normal_z.to(self.device)

            original_x = self.vae_model.decoder_forward(normal_z)
            original_dataset = original_x.detach().cpu().numpy()

            processed_z_by_first_actor = self.first_gan_model_actor(normal_z)
            processed_z_by_first_actor = self.vae_model.decoder_forward(processed_z_by_first_actor)
            processed_half_dataset = processed_z_by_first_actor.detach().cpu().numpy()

            processed_z_by_first_actor = self.first_gan_model_actor(normal_z)
            processed_z_by_second_actor = self.actor(processed_z_by_first_actor)
            processed_z_by_second_actor= self.vae_model.decoder_forward(processed_z_by_second_actor)
            processed_completely_dataset = processed_z_by_second_actor.detach().cpu().numpy()

            np.save(os.path.join(self.sample_path, 'origin.npy'), original_dataset)
            np.save(os.path.join(self.sample_path, 'data_modify_once.npy'), processed_half_dataset)
            np.save(os.path.join(self.sample_path, 'data_modify_twice.npy'), processed_completely_dataset)
        print('[*] Finish generating sample dataset')
        pass

    def test(self):
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

                test_dataset[
                (count * self.test_batch_size):((count + 1) * self.test_batch_size)] = x_recon.detach().cpu().numpy()
                count += 1
                pbar_iteration.update(1)

        np.save(os.path.join(self.test_path, 'data_for_valid.npy'), test_dataset)
        pbar_iteration.write('[*] Finish generating test dataset')
        pbar_iteration.close()
        return

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

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
