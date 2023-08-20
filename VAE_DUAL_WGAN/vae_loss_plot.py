# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/15 22:47
import numpy as np
import matplotlib.pyplot as plt

def vae_loss_plot(loss_data):
    epoch = loss_data[:, 0][200:]
    kld = loss_data[:, 2][200:]
    position_loss = loss_data[:, 3][200:]
    velocity_loss = loss_data[:, 4][200:]
    total_loss = loss_data[:, -1][200:]

    plt.figure(3, figsize=(12, 8))
    loss = [kld, position_loss, velocity_loss, total_loss]
    titles = ['KLD', 'Pos_Loss', 'Veloc_Loss', 'Total_Loss']
    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(epoch, loss[i], color=colors[i])
        # plt.title(titles[i])
        plt.xlabel('Number of epoch')
        plt.ylabel('{0}'.format(titles[i]))
    plt.subplots_adjust(left=0.08, bottom=0.125, right=0.98, top=0.88, hspace=0.2, wspace=0.2)
    plt.savefig('./processed_by_zwt/image/vae_loss_without_3std.svg', dpi=600)
    # plt.show()
    plt.close('all')

def vae_loss_plot_with_variance(loss_data, variance=True):
    epoch = loss_data[:, 0][209:35000:7] + 1
    kld = loss_data[:, 2][209:35000:7]
    position_loss = loss_data[:, 3][209:35000:7]
    velocity_loss = loss_data[:, 4][209:35000:7]
    total_loss = loss_data[:, -1][209:35000:7]

    plt.figure(3, figsize=(12, 8))
    loss = [kld, position_loss, velocity_loss, total_loss]
    titles = ['KLD', 'Position_Loss', 'Velocity_Loss', 'Total_Loss']
    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(epoch, loss[i], color=colors[i])
        if variance:
            # mean_loss = np.mean(loss[i])
            # variance_loss = np.var(loss[i])
            std = np.std(loss[i])
            y1 = loss[i] + 3 * std
            y2 = loss[i] - 3 * std
            plt.fill_between(epoch, y1, y2, alpha=0.3)

        plt.xlabel('Number of epoch')
        plt.ylabel('{0}'.format(titles[i]))
    # plt.title('Loss')
    plt.subplots_adjust(left=0.08, bottom=0.125, right=0.98, top=0.88, hspace=0.2, wspace=0.2)
    plt.savefig('./processed_by_zwt/image/vae_loss_with_3std.svg', dpi=600)
    # plt.show()
    plt.close('all')

def gan_loss_plot(loss_data):
    epoch = loss_data[:, 0][:]
    wasserstein_dis = loss_data[:, 2][:]
    critic_loss = loss_data[:, 3][:]
    actor_loss = loss_data[:, -1][:]

    plt.figure(3, figsize=(16, 6))
    loss = [wasserstein_dis, critic_loss, actor_loss]
    titles = ['Wasserstein_D', 'Critic_loss', 'Actor_loss']
    colors = ['red', 'blue', 'purple']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(epoch, loss[i], color=colors[i])
        plt.title(titles[i])
        plt.xlabel('Number of epoch')
        plt.ylabel('{0}'.format(titles[i]))
    plt.subplots_adjust(left=0.03, bottom=0.125, right=0.98, top=0.88, hspace=0.2, wspace=0.2)
    plt.show()
    # plt.savefig('./processed_by_zwt/image/gan_loss_without_3std.svg', dpi=600)
    plt.close('all')

def gan_loss_plot_with_variance(loss_data, variance=True):
    epoch = loss_data[:, 0][:]
    wasserstein_dis = loss_data[:, 2][:]
    critic_loss = loss_data[:, 3][:]
    actor_loss = loss_data[:, -1][:]

    plt.figure(3, figsize=(16, 6))
    loss = [wasserstein_dis, critic_loss, actor_loss]
    titles = ['Wasserstein_D', 'Critic_loss', 'Actor_loss']
    colors = ['red', 'blue', 'purple']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(epoch, loss[i], color=colors[i])
        if variance:
            std = np.std(loss[i])
            y1 = loss[i] + 3 * std
            y2 = loss[i] - 3 * std
            plt.fill_between(epoch, y1, y2, alpha=0.3)

        plt.xlabel('Number of epoch')
        plt.ylabel('{0}'.format(titles[i]))

    plt.title('GAN_Loss')
    plt.subplots_adjust(left=0.03, bottom=0.125, right=0.98, top=0.88, hspace=0.2, wspace=0.2)
    plt.savefig('./processed_by_zwt/image/gan_loss_with_3std.svg', dpi=600)
    plt.close('all')

def loss_plot(loss_data_vae, loss_data_e_wgan, loss_data_d_wgan):
    epoch_vae = loss_data_vae[:, 0][209:35000] + 1
    kld = loss_data_vae[:, 2][209:35000]
    position_loss = loss_data_vae[:, 3][209:35000]
    velocity_loss = loss_data_vae[:, 4][209:35000]
    total_loss = loss_data_vae[:, -1][209:35000]

    plt.figure(3, figsize=(18, 4))
    loss = [kld, position_loss, velocity_loss, total_loss]
    titles = ['KLD (VAE)', 'Position Loss (VAE)', 'Velocity Loss (VAE)', 'Total Loss (VAE)']
    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):
        plt.subplot(2, 5, i + 1)
        plt.plot(epoch_vae, loss[i], color=colors[i])
        # plt.title(titles[i])
        plt.xlabel('Number of epoch', fontsize=16)
        plt.ylabel('{0}'.format(titles[i]), fontsize=16)

    epoch_e_wgan = loss_data_e_wgan[:, 0][:]
    wasserstein_dis = loss_data_e_wgan[:, 2][:]
    critic_loss = loss_data_e_wgan[:, 3][:]
    actor_loss = loss_data_e_wgan[:, -1][:]

    loss = [wasserstein_dis, critic_loss, actor_loss]
    titles = ['Wasserstein D (E-WGAN)', 'Critic loss (E-WGAN)', 'Actor loss (E-WGAN)']
    colors = ['red', 'blue', 'purple']

    for i in range(4, 7):
        plt.subplot(2, 5, i + 1)
        plt.plot(epoch_e_wgan, loss[i-4], color=colors[i-4])
        # plt.title(titles[i-4])
        plt.xlabel('Number of epoch', fontsize=16)
        plt.xticks([0, 60000, 120000])
        plt.ylabel('{0}'.format(titles[i-4]), fontsize=16)

    epoch_d_wgan = loss_data_d_wgan[:, 0][:]
    wasserstein_dis = loss_data_d_wgan[:, 2][:]
    critic_loss = loss_data_d_wgan[:, 3][:]
    actor_loss = loss_data_d_wgan[:, -1][:]

    loss = [wasserstein_dis, critic_loss, actor_loss]
    titles = ['Wasserstein D (D-WGAN)', 'Critic loss (D-WGAN)', 'Actor loss (D-WGAN)']
    colors = ['red', 'blue', 'purple']

    for i in range(7, 10):
        plt.subplot(2, 5, i + 1)
        plt.plot(epoch_d_wgan, loss[i-7], color=colors[i-7])
        # plt.title(titles[i-7])
        plt.xlabel('Number of epoch', fontsize=16)
        plt.xticks([0, 60000, 120000])
        plt.ylabel('{0}'.format(titles[i-7]), fontsize=16)

    plt.subplots_adjust(left=0.02, bottom=0.08, right=0.98, top=0.98, hspace=0.3, wspace=0.3)
    plt.savefig('./processed_by_zwt/image/loss_plot.svg', dpi=600)
    plt.show()
    plt.close('all')
    pass

if __name__ == '__main__':
    vae_loss_data = np.load('../naive_VAE/processed_by_zwt/loss/loss0.npy', allow_pickle=True)
    gan_loss_data_1 = np.load('../VAE_WGAN/processed_by_zwt/loss/loss0.npy', allow_pickle=True)
    gan_loss_data_2 = np.load('./processed_by_zwt/loss/loss1.npy', allow_pickle=True)
    gan_loss_data_3 = np.load('./processed_by_zwt/loss/loss3.npy', allow_pickle=True)
    print(vae_loss_data.shape)
    # vae_loss_plot(vae_loss_data)
    # vae_loss_plot_with_variance(vae_loss_data)
    # gan_loss_plot(gan_loss_data_1)
    # gan_loss_plot_with_variance(gan_loss_data)
    # gan_loss_plot(gan_loss_data_3)
    loss_plot(vae_loss_data, gan_loss_data_1, gan_loss_data_2)

