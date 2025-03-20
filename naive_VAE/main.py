import argparse
import torch
import os

from trainer import VAE_Trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    beta = [0.001]
    gama = [50]
    cita = [10]
    for i in range(1):
        parser = argparse.ArgumentParser()

        parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')
        parser.add_argument('-type', default=1, type=int, help='type of stage, 0 means train, 1 means sample, 2 means test')
        parser.add_argument('-id', default=i+1, type=int, help='index of the model')
        parser.add_argument('-max_iteration', default=20000, type=int, help='maximum training iteration')
        parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')
        parser.add_argument('-print_iter', default=1, type=int, help='print losses iter')
        parser.add_argument('-save_epoch', default=10, type=int, help='the iteration that save models')
        parser.add_argument('-num_workers', default=0, type=int, help='dataloder num_workers')

        parser.add_argument('-z_dim', default=64, type=int, help='dimension of the representation z')
        parser.add_argument('-scene_len', default=125, type=int, help='length for a single scenario')
        parser.add_argument('-scene_dim', default=6, type=int, help='dimensions contained in scenarios')
        parser.add_argument('-lr', default=1e-4, type=float, help='base learning rate of the model')
        parser.add_argument('-alpha', default=1, type=float, help='weight of position reconstruction error')
        parser.add_argument('-beta', default=beta[i], type=float, help='weight of kl divergence')
        parser.add_argument('-gama', default=gama[0], type=float, help='weight of velocity reconstruction error')

        parser.add_argument('-training_data_path', default='./processed/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', type=str, help='path to the vae model training data')  # 处理后的归一化后的训练集位置
        parser.add_argument('-testing_data_path', default='./processed/input_data/cutin3_xy_singleD_proceseed_norm_test.npy', type=str, help='path to the vae model testing data')

        parser.add_argument('-save_model_path', default='./processed/saved-model', type=str, help='rootpath for saving vae model')
        parser.add_argument('-sample_path', default='./processed/samples', type=str, help='path to the vae generated data')
        parser.add_argument('-test_path', default='./processed/test', type=str, help='path to the vae test data')
        parser.add_argument('-loss_path', default='./processed/loss', type=str, help='path to the gan loss')

        parser.add_argument('-weight', default=[1 * cita[0], 1, 1 * cita[0], 1, 1 * cita[0], 1], type=int, help='weight of each scenario dimension')
        parser.add_argument('-latent_cons', default=False, type=bool, help='whether to constrain the latent parameters')
        parser.add_argument('-batch_size', default=512, type=int, help='batch size')
        parser.add_argument('-test_batch_size', default=2000, type=int, help='batch size for test')
        parser.add_argument('-sample_batch_size', default=1000, type=int, help='batch size for sample')

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        net = VAE_Trainer(args)

        if args.type == 0:
            net.train()
        elif args.type == 1:
            net.sample()
        elif args.type == 2:
            net.test()
        else:
            print('No match type')