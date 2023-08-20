"""
@Author: Syh
@Date: 2022-12-24 14:29
@Description: 针对VAE+GAN联合泛化，训练VAE模型，该模型应具有强重建能力（牺牲泛化能力）
@Reference: Latent-Constraints-Learning-to-Generate-Conditionally-from-Unconditional-Generative-Models
"""

import argparse
import torch
import os

from trainer import VAE_Trainer

# 提升运行效率的设置——非确定性算法，以牺牲可复现为代价
# 一般在开始前都会设置以下两行
torch.backends.cudnn.enabled = True  # 使用非确定性算法
torch.backends.cudnn.benchmark = True  # 自动搜寻适合当前配置的最高效算法
torch.backends.cudnn.deterministic = True  # 避免随机性结果波动_by


if __name__ == "__main__":
    beta = [0.1]  # KLD散度权重
    gama = [50]  # 纵向速度权重
    cita = [10]  # 位置权重
    for i in range(1):
        parser = argparse.ArgumentParser()

        parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')  # 是否继续训练
        parser.add_argument('-type', default=2, type=int, help='type of stage, 0 means train, 1 means sample, 2 means test')  # 训练模式
        parser.add_argument('-id', default=i, type=int, help='index of the model')  # 模型保存时的id
        parser.add_argument('-max_iteration', default=20000, type=int, help='maximum training iteration')  # 最大的迭代次数（这里的max_epoch使用不准确）
        parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')  # cuda的编号
        parser.add_argument('-print_iter', default=1, type=int, help='print losses iter')  # 每隔几个iter输出损失函数
        parser.add_argument('-save_epoch', default=10, type=int, help='the iteration that save models')  # 每个几个iter保存模型
        parser.add_argument('-num_workers', default=0, type=int, help='dataloder num_workers')  # 进程数

        parser.add_argument('-z_dim', default=64, type=int, help='dimension of the representation z')  # 隐变量维度
        parser.add_argument('-scene_len', default=125, type=int, help='length for a single scenario')  # 场景时序长度
        parser.add_argument('-scene_dim', default=6, type=int, help='dimensions contained in scenarios')  # 场景维度
        parser.add_argument('-lr', default=1e-4, type=float, help='base learning rate of the model')  # 学习率
        parser.add_argument('-alpha', default=1, type=float, help='weight of position reconstruction error')  # 重建误差中纵向位置的权重
        parser.add_argument('-beta', default=beta[i], type=float, help='weight of kl divergence')  # 泛化误差
        parser.add_argument('-gama', default=gama[0], type=float, help='weight of velocity reconstruction error')  # 重建误差中速度项权重

        parser.add_argument('-training_data_path', default='./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', type=str, help='path to the vae model training data')  # 处理后的归一化后的训练集位置
        parser.add_argument('-testing_data_path', default='./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_test.npy', type=str, help='path to the vae model testing data')

        parser.add_argument('-save_model_path', default='./processed_by_zwt/saved-model', type=str, help='rootpath for saving vae model')  # 保存模型的位置
        parser.add_argument('-sample_path', default='./processed_by_zwt/samples', type=str, help='path to the vae generated data')  # 保存sample样本的位置
        parser.add_argument('-test_path', default='./processed_by_zwt/test', type=str, help='path to the vae test data')  # 测试过程中重建数据的保存位置
        parser.add_argument('-loss_path', default='./processed_by_zwt/loss', type=str, help='path to the gan loss')  # 训练过程中损失函数保存位置

        parser.add_argument('-weight', default=[1 * cita[0], 1, 1 * cita[0], 1, 1 * cita[0], 1], type=int, help='weight of each scenario dimension')
        parser.add_argument('-latent_cons', default=False, type=bool, help='whether to constrain the latent parameters')  # 是否进行隐变量限制
        parser.add_argument('-batch_size', default=512, type=int, help='batch size')  # batch_size
        parser.add_argument('-test_batch_size', default=2000, type=int, help='batch size for test')  # 测试集batch_size
        parser.add_argument('-sample_batch_size', default=3500, type=int, help='batch size for sample')  # 采样一次采多少个

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)   # gpu上运行
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 解决unspecified launch failure
        net = VAE_Trainer(args)  # 实例化

        if args.type == 0:
            net.train()
        elif args.type == 1:
            net.sample()
        elif args.type == 2:
            net.test()
        else:
            print('No match type')
