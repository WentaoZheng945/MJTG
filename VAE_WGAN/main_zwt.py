'''
@Author: Syh
@Date: 2022-12-18 14:25
@Description: 针对VAE+GAN泛化，尝试融入GAN对VAE的隐空间采样过程进行约束，提升泛化质量
@Reference: Latent-Constraints-Learning-to-Generate-Conditionally-from-Unconditional-Generative-Models
@Update: 2023-01-03 将GAN框架，更新为WGAN，以解决GAN训练不稳定及模式崩溃问题
'''

import argparse
import torch
import os

from trainer import AC_Trainer

# 提升运行效率的设置——非确定性算法，以牺牲可复现为代价
torch.backends.cudnn.enabled = True  # 使用非确定性算法
torch.backends.cudnn.benchmark = True  # 自动搜寻适合当前配置的最高效算法
torch.backends.cudnn.deterministic = True  # 避免随机性结果波动_by


if __name__ == "__main__":
    c = [0.01, 0.05, 0.1]
    num = [2, 3, 4]
    for i in range(1):
        parser = argparse.ArgumentParser()

        parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')
        parser.add_argument('-type', default=2, type=int, help='type of stage, 0 means train, 1 means sample, 2 means test')
        parser.add_argument('-id', default=0, type=int, help='index of the model')
        parser.add_argument('-max_epoch', default=150000, type=int, help='maximum training epoch')
        parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')
        parser.add_argument('-batch_size', default=512, type=int, help='batch size')
        parser.add_argument('-sample_batch_size', default=3500, type=int, help='batch size for test')
        parser.add_argument('-print_iter', default=100, type=int, help='print losses iter')
        parser.add_argument('-save_epoch', default=1000, type=int, help='the iteration that save models')
        parser.add_argument('-num_workers', default=0, type=int, help='dataloder num_workers')  # 进程

        parser.add_argument('-z_dim', default=64, type=int, help='dimension of the representation z')  # 隐变量维度
        parser.add_argument('-scene_len', default=125, type=int, help='length for a single scenario')  # 时长
        parser.add_argument('-scene_dimension', default=6, type=int, help='dimensions contained in scenarios')  # 特征维度
        parser.add_argument('-num_layer', default=2, type=int, help='number of layer in G&D')  # 生成器和判别器的层数
        parser.add_argument('-lr', default=5e-5, type=float, help='base learning rate of the model')  # 学习率
        parser.add_argument('-distance_penalty', default=0.1, type=float, help='weigiht of the regularization term')  # 正则项惩罚权重
        parser.add_argument('-weight_cliping_limit', default=0.05, type=float, help='CP for WGAN')  # WGAN网络参数截断范围

        parser.add_argument('-vae_model_path', default='../naive_VAE/processed_by_zwt/saved-model', type=str, help='rootpath for saving vae model')  # VAE模型存放位置
        parser.add_argument('-vae_model_id', default=1, type=int, help='index of the vae model')  # 训练好的VAE模型编号
        parser.add_argument('-save_model_path', default='./processed_by_zwt/saved-model', type=str, help='rootpath for saving gan model')  # 保存训练后的GAN模型位置

        parser.add_argument('-training_data_path', default='../naive_VAE/processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', type=str, help='path to the vae model training data')
        parser.add_argument('-testing_data_path', default='../naive_VAE/processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_test.npy', type=str, help='path to the vae model testing data')
        parser.add_argument('-sample_path', default='./processed_by_zwt/samples', type=str, help='path to the gan generated data')
        parser.add_argument('-loss_path', default='./processed_by_zwt/loss', type=str, help='path to the gan loss')
        parser.add_argument('-test_path', default='./processed_by_zwt/test', type=str, help='path to the reconstruction data')

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 解决unspecified launch failure
        net = AC_Trainer(args)

        if args.type == 0:
            net.train()  # 0训练
        elif args.type == 1:
            net.sample()  # 1采样
        elif args.type == 2:
            print(1)
            net.test()  # 2测试
        else:
            print('No match type')
