# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/12 22:11
import matplotlib.pyplot as plt
import numpy as np
import os
from cutin_evaluation import Repair_cutin
from subFunctions import cutinPloter


def reconstruction_evaluation(lims, testScene_path, reconstruction_path):
    testScene = Repair_cutin(fusion_path=testScene_path, lims=lims, isNorm=True, need_repair=False)
    reconstruction = Repair_cutin(fusion_path=reconstruction_path, lims=lims, isNorm=True, need_repair=False)
    eval = reconstruction.reconEval(testScene.fusion_xy)
    print('ADE: {}, FDE: {}'.format(str(eval[0]), str(eval[1])))
    pass

def generation_evaluation(lims, trainScene_path, sampleScene_without_latent_path, sampleScene_with_latent_path):
    trainScene = Repair_cutin(fusion_path=trainScene_path, lims=lims, isNorm=True, need_repair=False)
    sampleScene_without_latent = Repair_cutin(fusion_path=sampleScene_without_latent_path, lims=lims, isNorm=True, need_repair=False)
    sampleScene_with_latent = Repair_cutin(fusion_path=sampleScene_with_latent_path, lims=lims, isNorm=True, need_repair=False)
    pass

def trajectory_plot_ego_and_adversary(case_scene, scenario_id):
    direction_list = ['_longitude', '_latitude']
    time = np.arange(0, 5, 0.04)
    for i in range(4):
        if i == 0:
            label = 'ego' + direction_list[0]
        elif i == 1:
            label = 'ego' + direction_list[1]
        elif i == 2:
            label = 'adversary' + direction_list[0]
        else:
            label = 'adversary' + direction_list[1]
        title = str(scenario_id) + '_' + str(label)

        # 第一张子图:累计位移
        plt.subplot(3, 1, 1)  # 累计位移
        plt.plot(time, case_scene.fusion_xy[scenario_id, :, i], '--k', linewidth=0.25, label=label+'_original')  # 实线 * 黑色 *大小1.5 线宽2.5
        plt.plot(time, case_scene.fusion_xy_repair[scenario_id, :, i], '-*g', linewidth=0.25, label=label+str('_enhanced'), markersize=0.5)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')  # 纵坐标为横向位移
        # plt.title(title)

        # 第二张子图:速度
        plt.subplot(3, 1, 2)
        plt.plot(time, case_scene.fusion_v[scenario_id, :, i], '--k', linewidth=0.25, label=label+'_original')  # 实线 * 黑色 *大小1.5 线宽2.5
        plt.plot(time, case_scene.fusion_v_repair[scenario_id, :, i], '-*g', linewidth=0.25, label=label + str('_enhanced'), markersize=0.5)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')  # 纵坐标为速度
        if 'longitude' in label:
            plt.ylim(0, 50)
        else:
            plt.ylim(-10, 10)

        # 第三张子图:加速度
        plt.subplot(3, 1, 3)
        plt.plot(time, case_scene.fusion_a[scenario_id, :, i], '--k', linewidth=0.25, label=label + str('_original'))  # 实线 * 黑色 *大小1.5 线宽2.5
        plt.plot(time, case_scene.fusion_a_repair[scenario_id, :, i], '-*g', linewidth=0.25, label=label + str('_enhanced'), markersize=0.5)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')  # 纵坐标为加速度
        if 'longitude' in label:
            plt.ylim(-12, 12)
        else:
            plt.ylim(-5, 5)

        # 保存位置
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        save_path = r'processed/image/case_analysis/' + str(scenario_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = save_path + '/' + str(label) + '.svg'
        plt.savefig(save_path, dpi=600)
        plt.close('all')
    pass

def case_analysis(lims, case_path):
    caseScene = Repair_cutin(fusion_path=case_path, lims=lims, isNorm=True, need_repair=True)
    # 绘制主车及对手车的位置、速度、加速度图像
    for i in range(caseScene.fusion_xy.shape[0]):
        trajectory_plot_ego_and_adversary(caseScene, i)
    # 绘制动态cut in过程
    # cutinPloter(caseScene.fusion_xy_repair)
    pass

def calc_rate_of_failure(lims, sampleScene_with_latent_path):
    sampleScene = Repair_cutin(fusion_path=sampleScene_with_latent_path, lims=lims, isNorm=True, need_repair=True)
    pass


if __name__ == '__main__':
    x1_min, x1_max = 0.0, 235.5454259747434
    y1_min, y1_max = -3.679218247347115, 2.5978529242443447
    x2_min, x2_max = -19.260000000000048, 385.4497701569932
    y2_min, y2_max = -5.030000000000001, 6.909999999999997
    x3_min, x3_max = 8.329999999999998, 413.4503592184133
    y3_min, y3_max = -3.460082480103437, 3.69
    lims = [x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max, x3_min, x3_max, y3_min, y3_max]

    # 定义场景的输入
    global_trainScene_path = r'./processed/input_data/cutin3_xy_singleD_proceseed_norm_train.npy'
    global_testScene_path = r'./processed/input_data/cutin3_xy_singleD_proceseed_norm_test.npy'
    global_reconstruction_path = r'./processed/test/data_for_valid.npy'
    global_sampleScene_without_latent_path = r'./processed/samples/data_modify_once.npy'
    global_sampleScene_with_latent_path = r'./processed/samples/data_modify_twice.npy'
    global_sampleScene_for_plot_path = r'./processed/samples/cutin3_sample_for_plot.npy'

    # 重建能力评价
    # reconstruction_evaluation(lims, global_testScene_path, global_reconstruction_path)

    # 泛化能力评价
    # generation_evaluation(lims, global_trainScene_path, global_sampleScene_without_latent_path, global_sampleScene_with_latent_path)

    # 案例分析，绘制图像
    # case_analysis(lims, global_sampleScene_for_plot_path)

    # 计算一下未通过率
    calc_rate_of_failure(lims, global_sampleScene_with_latent_path)




