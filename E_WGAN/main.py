import argparse
import torch
import os

from trainer import AC_Trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    c = [0.01, 0.05, 0.1]
    num = [2, 3, 4]
    for i in range(1):
        parser = argparse.ArgumentParser()

        parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')
        parser.add_argument('-type', default=0, type=int, help='type of stage, 0 means train, 1 means sample')
        parser.add_argument('-id', default=0, type=int, help='index of the model')
        parser.add_argument('-max_epoch', default=150000, type=int, help='maximum training epoch')
        parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')
        parser.add_argument('-batch_size', default=512, type=int, help='batch size')
        parser.add_argument('-sample_batch_size', default=3500, type=int, help='batch size for test')
        parser.add_argument('-print_iter', default=100, type=int, help='print losses iter')
        parser.add_argument('-save_epoch', default=1000, type=int, help='the iteration that save models')
        parser.add_argument('-num_workers', default=0, type=int, help='dataloder num_workers')

        parser.add_argument('-z_dim', default=64, type=int, help='dimension of the representation z')
        parser.add_argument('-scene_len', default=125, type=int, help='length for a single scenario')
        parser.add_argument('-scene_dimension', default=6, type=int, help='dimensions contained in scenarios')
        parser.add_argument('-num_layer', default=2, type=int, help='number of layer in G&D')
        parser.add_argument('-lr', default=5e-5, type=float, help='base learning rate of the model')
        parser.add_argument('-distance_penalty', default=0.1, type=float, help='weigiht of the regularization term')
        parser.add_argument('-weight_cliping_limit', default=0.05, type=float, help='CP for WGAN')

        parser.add_argument('-vae_model_path', default='../naive_VAE/processed/saved-model', type=str, help='rootpath for saving vae model')
        parser.add_argument('-vae_model_id', default=1, type=int, help='index of the vae model')
        parser.add_argument('-save_model_path', default='./processed/saved-model', type=str, help='rootpath for saving gan model')

        parser.add_argument('-training_data_path', default='../naive_VAE/processed/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', type=str, help='path to the vae model training data')
        parser.add_argument('-testing_data_path', default='../naive_VAE/processed/input_data/cutin3_xy_singleD_proceseed_norm_test.npy', type=str, help='path to the vae model testing data')
        parser.add_argument('-sample_path', default='./processed/samples', type=str, help='path to the gan generated data')
        parser.add_argument('-loss_path', default='./processed/loss', type=str, help='path to the gan loss')

        args = parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        net = AC_Trainer(args)

        if args.type == 0:
            net.train()
        elif args.type == 1:
            net.sample()
        else:
            print('No match type')
