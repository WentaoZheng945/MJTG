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
    plt.savefig('./processed/image/vae_loss_without_3std.svg', dpi=600)
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
    plt.savefig('./processed/image/vae_loss_with_3std.svg', dpi=600)
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
    plt.savefig('./processed/image/gan_loss_without_3std.svg', dpi=600)
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
    plt.savefig('./processed/image/gan_loss_with_3std.svg', dpi=600)
    plt.close('all')

if __name__ == '__main__':
    vae_loss_data = np.load('../naive_VAE/processed/wloss/loss1.npy', allow_pickle=True)
    gan_loss_data_1 = np.load('../VAE_WGAN/processed/loss/loss0.npy', allow_pickle=True)
    gan_loss_data_2 = np.load('./processed/loss/loss0.npy', allow_pickle=True)
    print(vae_loss_data.shape)
    # vae_loss_plot(vae_loss_data)
    # vae_loss_plot_with_variance(vae_loss_data)
    # gan_loss_plot(gan_loss_data_1)
    # gan_loss_plot_with_variance(gan_loss_data)
    gan_loss_plot(gan_loss_data_2)
