# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/11 20:43
import numpy as np
import matplotlib.pyplot as plt

def loss_plot(loss_data):
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

def loss_plot_with_variance(loss_data, variance=True):
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


loss_data = np.load('./processed_by_zwt/loss/loss0.npy', allow_pickle=True)
print(loss_data.shape)
loss_plot(loss_data)
loss_plot_with_variance(loss_data)
