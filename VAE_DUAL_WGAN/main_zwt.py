# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/18 17:02
import argparse
import torch
import os
from trainer import AC_Trainer_2


# 提升运行效率的设置——非确定性算法，以牺牲可复现为代价
torch.backends.cudnn.enabled = True  # 使用非确定性算法
torch.backends.cudnn.benchmark = True  # 自动搜寻适合当前配置的最高效算法
torch.backends.cudnn.deterministic = True  # 避免随机性结果波动_by

"""
在VAE+WGAN(合理性修正)基础上加入第二个WGAN(向着危险定向修正)，提升泛化样本中危险场景的数量
"""

if __name__ == '__main__':
    for i in range(1):
        parser = argparse.ArgumentParser()

        parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')
        parser.add_argument('-type', default=1, type=int, help='type of stage, 0 means train, 1 means sample, 2 means test')  # 此模型启动模式
        parser.add_argument('-id', default=3, type=int, help='index of the model')  # 此WGAN模型的id
        parser.add_argument('-max_epoch', default=150000, type=int, help='maximum training epoch')  # 最大epoch
        parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')  # gpu编号
        parser.add_argument('-batch_size_danger', default=160, type=int, help='batch size danger')  # 训练中的批大小
        parser.add_argument('-batch_size_safe', default=512, type=int, help='batch size safe')  # 训练中的批大小
        parser.add_argument('-sample_batch_size', default=3500, type=int, help='batch size for sample')  # 采样多少
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
        parser.add_argument('-first_gan_model_path', default='../VAE_WGAN/processed_by_zwt/saved-model', type=str, help='rootpath for saving first gan model')  # GAN模型存放位置
        parser.add_argument('-first_gan_model_id', default=0, type=int, help='index of the first gan model')  # 训练好的GAN模型编号
        parser.add_argument('-save_model_path', default='./processed_by_zwt/saved-model', type=str, help='rootpath for saving second gan model')  # 保存此次训练后的GAN模型位置

        parser.add_argument('-training_data_danger_path', default='./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train_danger_3.npy', type=str, help='path to the gan model danger training data')
        parser.add_argument('-training_data_safe_path', default='./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train_safe_3.npy', type=str, help='path to the gan model safe training data')
        parser.add_argument('-training_data_path', default='./naive_VAE/processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', type=str, help='path to the vae model training data')
        parser.add_argument('-testing_data_path', default='../naive_VAE/processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_test.npy', type=str, help='path to the vae model testing data')
        parser.add_argument('-sample_path', default='./processed_by_zwt/samples', type=str, help='path to the gan generated data')  # 保存sample数据的位置
        parser.add_argument('-loss_path', default='./processed_by_zwt/loss', type=str, help='path to the gan loss')
        parser.add_argument('-test_path', default='./processed_by_zwt/test', type=str, help='path to the reconstruction data')  # 保存重建数据的位置

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 解决unspecified launch failure
        net = AC_Trainer_2(args)

        if args.type == 0:
            net.train()  # 0训练
        elif args.type == 1:
            net.sample()  # 1采样
        elif args.type == 2:
            net.test()  # 2测试
        else:
            print('No match type')