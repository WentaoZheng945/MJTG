import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import CUDA


class VAE(nn.Module):
    def __init__(self, z_dim, scene_len, scene_dim):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.seq_dim = scene_dim
        self.seq_len = scene_len

        self.input_size = 64 * 2
        self.num_layers = 1
        self.hidden_size_encoder = 128 * 2
        self.hidden_size_decoder = 128 * 2

        # 1. build encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(self.seq_dim, self.input_size),
            nn.ReLU())

        self.encoder_gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_encoder, num_layers=self.num_layers, batch_first=True, bidirectional=True)

        self.encoder = nn.Linear(2 * self.hidden_size_encoder, 2 * z_dim)

        # 2. build decoder
        self.z2hidden = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_size_decoder),
            nn.ReLU())

        self.hidden2input = nn.Sequential(
            nn.Linear(self.hidden_size_decoder, int(self.input_size / 2)),
            nn.ReLU())

        self.decoder_gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size_decoder, num_layers=self.num_layers, batch_first=True)

        self.embedding_decoder = nn.Sequential(
            nn.Linear(self.seq_dim, int(self.input_size / 2)),
            nn.ReLU())

        self.decoder = nn.Linear(self.hidden_size_decoder, self.seq_dim)

    def encoder_forward_(self, x):
        batch_size = x.size(0)  # [batch_size, 125, 6]
        x_embedding = self.embedding_encoder(x)  # [batch_size, 125, 128]

        # GRU
        hidden_encoder = self.init_hidden(batch_size, self.hidden_size_encoder)
        output, _ = self.encoder_gru(x_embedding, hidden_encoder)  # [batch_size, 125, 512]
        last_output = self.encoder(output[:, -1, :])  # [batch_size, 2 * self.z_dim]

        # output
        mu = last_output[:, :self.z_dim]  # [batch_size, 64]
        logvar = last_output[:, self.z_dim:]  # [batch_size, 64]
        z = self.reparametrize(mu, logvar)  # [batch_size, 64]
        return z, [mu, logvar]

    def decoder_forward(self, z):
        batch_size = z.size(0)

        start_point = CUDA(Variable(torch.zeros(batch_size, 1, self.seq_dim)))

        # z: [batch_size, 64] -> [batch_size, 1, 64] | output: [batch_size, 1, 256] -> [1, batch_size, 256]
        hidden_decoder = self.z2hidden(z.view(batch_size, 1, self.z_dim)).permute(1, 0, 2)
        hidden = self.hidden2input(hidden_decoder.permute(1, 0, 2))  # [batch_size, 1, 64]

        # start_point: [batch_size, 1, 4]
        current_stage = self.embedding_decoder(start_point)  # [batch_size, 1, 64]
        start_embedding = torch.cat((current_stage, hidden), dim=2)

        x_recon = []
        for _ in range(self.seq_len):
            # output: [batch_size, 1, 256] hidden_decoder: [1, batch_size, 256]
            output, hidden_decoder = self.decoder_gru(start_embedding, hidden_decoder)  # [batch_size, 1, 256]
            # output: [batch_size, 256] -> [batch_size, 6] -> x_cur: [batch_size, 1, 6]
            x_cur = self.decoder(output[:, -1, :]).view(batch_size, -1, self.seq_dim)
            x_recon.append(x_cur.view(batch_size, -1))  # x_recon [[batch_size, 6]]

            current_stage = self.embedding_decoder(x_cur)  # x[t-1]: [batch_size, 1, 6]
            hidden = self.hidden2input(hidden_decoder.permute(1, 0, 2))  # h[t-1]: [batch_size, 1, 64]
            start_embedding = torch.cat((current_stage, hidden), dim=2)
        x_recon = torch.stack(x_recon, dim=0).permute(1, 0, 2)  # [batch_size, seq_len, seq_dim]
        return x_recon

    def forward(self, x):
        z, z_bag = self.encoder_forward_(x)  # z_bag:[mu, logvar]   mu:[batch_size, 32] logvar:[batch_size, 32]
        x_recon = self.decoder_forward(z)
        return x_recon, z_bag

    def init_hidden(self, batch_size, hidden_size):
        hidden_h = Variable(torch.zeros(self.num_layers * 2, batch_size, hidden_size))
        if torch.cuda.is_available():
            hidden_h = hidden_h.cuda()
        return hidden_h

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = CUDA(Variable(std.data.new(std.size()).normal_()))
        return mu + std * eps