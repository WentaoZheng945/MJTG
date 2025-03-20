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
        self.continue_training = args.continue_training
        self.model_id = args.id
        self.max_iteration = args.max_iteration
        self.print_iter = args.print_iter
        self.save_epoch = args.save_epoch
        self.num_workers = args.num_workers

        self.z_dim = args.z_dim
        self.scene_len = args.scene_len
        self.scene_dim = args.scene_dim
        self.lr = args.lr
        self.alpha = args.alpha
        self.beta = args.beta
        self.gama = args.gama
        self.weight = args.weight

        self.training_data_path = args.training_data_path
        self.testing_data_path = args.testing_data_path

        self.latent_cons = args.latent_cons
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.sample_batch_size = args.sample_batch_size

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # checkpoint
        self.save_model_path = args.save_model_path
        mkdirs(self.save_model_path)

        self.sample_path = args.sample_path
        mkdirs(self.sample_path)

        self.test_path = args.test_path
        mkdirs(self.test_path)

        self.loss_path = args.loss_path
        mkdirs(self.loss_path)

        # set seed
        set_seed(1)

        # model
        self.vae = CUDA(VAE(z_dim=self.z_dim, scene_len=self.scene_len, scene_dim=self.scene_dim))
        self.optimizer = optim.Adam(params=self.vae.parameters(), lr=3 * self.lr)  # TODO 这里为啥学习率乘3
        if self.continue_training:
            self.load_model()
            print('[*] Finish loading parameters from file')
        else:
            self.vae.apply(kaiming_init)
            print('[*] Finish building model')

        # data
        self.x = CUDA(Variable(torch.FloatTensor(self.batch_size, self.scene_len, self.scene_dim)))  # TODO

        # load data
        self.dataset = Sequence_Dataset(self.training_data_path)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('[*] Finish loading dataset')

    def train(self):
        loss_collector = []
        pbar_epoch = tqdm(total=self.max_iteration, desc='[Epoch]')
        max_iteration = int(len(self.dataset) / self.batch_size)
        self.vae.train()
        for epoch in range(self.max_iteration):
            pbar_iteration = tqdm(total=max_iteration, desc='[Iteration]')
            iteration = 0
            for x in self.dataloader:
                self.batch_size = x.size(0)
                self.x.resize_(self.batch_size, self.scene_len, self.scene_dim)

                pbar_iteration.update(1)
                iteration += 1
                self.x.copy_(x)

                x_recon, z_bag = self.vae(self.x)

                # kl divergence
                kld = kld_loss(z_bag)

                # reconstruction loss
                recon = [0] * self.x.shape[-1]
                for i in range(self.x.shape[-1]):
                    recon[i] = mse_loss(self.x[:, :, i], x_recon[:, :, i]) * self.weight[i]
                recon_loss = sum(recon)

                # velocity loss
                velocity = torch.diff(self.x, dim=1) / 0.04
                velocity_recon = torch.diff(x_recon, dim=1) / 0.04
                velocity_loss = mse_loss(velocity[:, :, 0], velocity_recon[:, :, 0]) + mse_loss(velocity[:, :, 2], velocity_recon[:, :, 2]) + mse_loss(velocity[:, :, 4], velocity_recon[:, :, 4])

                total_loss = recon_loss * self.alpha + kld * self.beta + velocity_loss * self.gama

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
        # load model
        self.load_model()
        print('[*] Finish loading parameters from file')

        dataset = Sequence_Dataset(self.testing_data_path)
        self.test_batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)
        test_dataset = np.zeros((len(dataset), self.scene_len, self.scene_dim))
        pbar_iteration = tqdm(total=int(len(dataset) / self.test_batch_size), desc='[Iteration]')
        count = 0
        self.vae.eval()
        with torch.no_grad():
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
            if not self.latent_cons:
                sample_z = CUDA(torch.randn(self.sample_batch_size, self.z_dim))
                sample_x = self.vae.decoder_forward(sample_z)
                sample_dataset = sample_x.detach().cpu().numpy()
                np.save(os.path.join(self.sample_path, 'data_without_latent_cons.npy'), sample_dataset)
        print('[*] Finish generating sample dataset')

    def save_model(self):
        states = {'vae_states': self.vae.state_dict(), 'optim_states': self.optimizer.state_dict()}

        filepath = os.path.join(self.save_model_path, 'model.' + str(self.model_id) + '.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.save_model_path, 'model.' + str(self.model_id) + '.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)

            self.vae.load_state_dict(checkpoint['vae_states'])
            self.optimizer.load_state_dict(checkpoint['optim_states'])
