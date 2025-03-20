# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/11 20:43
import numpy as np
import matplotlib.pyplot as plt

def loss_plot(loss_data):
    epoch = loss_data[:, 0]
    kld = loss_data[:, 2]
    position_loss = loss_data[:, 3]
    velocity_loss = loss_data[:, 4]
    total_loss = loss_data[:, -1]

    plt.figure(3, figsize=(20, 6))
    loss = [kld, position_loss, velocity_loss, total_loss]
    titles = ['KLD', 'Pos_Loss', 'Veloc_Loss', 'Total_Loss']
    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(epoch, loss[i], color=colors[i])
        plt.title(titles[i])
    plt.subplots_adjust(left=0.03, bottom=0.125, right=0.98, top=0.88, hspace=0.2, wspace=0.3)


loss_data = np.load('./processed/loss/loss0.npy')
loss_plot(loss_data)
plt.show()