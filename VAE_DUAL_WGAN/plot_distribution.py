# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/17 16:23
import matplotlib.pyplot as plt
import numpy as np
import os
from cutin_evaluation import Repair_cutin
from subFunctions import cutinPloter
import seaborn as sns
import torch.nn.functional as F
import torch
from math import log

def plot_distribution(real_world_scenarios, derived_scenarios):
    # 绘制MTTC分布
    ax1 = plt.subplot(111)
    input_mttc = real_world_scenarios.fusion_mttc
    sample_mttc = derived_scenarios.fusion_mttc
    sns.kdeplot(data=input_mttc, label="real-world Scenarios", fill=True, ax=ax1)
    sns.kdeplot(data=sample_mttc, label="derived Scenarios", fill=True, ax=ax1)
    ax1.legend()
    ax1.set_ylabel('Density')
    ax1.set_xlabel('MTTC (s)')
    plt.savefig('./processed_by_zwt/image/minMTTC_distribution.svg', dpi=600)
    # 绘制THW分布
    plt.clf()
    ax2 = plt.subplot(111)
    input_thw = real_world_scenarios.fusion_thw
    sample_thw = derived_scenarios.fusion_thw
    sns.kdeplot(data=input_thw, label="real-world Scenarios", fill=True, ax=ax2)
    sns.kdeplot(data=sample_thw, label="derived Scenarios", fill=True, ax=ax2)
    ax2.legend()
    ax2.set_ylabel('Density')
    ax2.set_xlabel('THW (s)')
    plt.tight_layout()
    plt.savefig('./processed_by_zwt/image/minTHW_distribution.svg', dpi=600)
    plt.clf()
    return

def plot_hist(real_world_scenarios, derived_scenarios, derived_2_scenarios, derived_3_scenarios, derived_4_scenarios):
    plt.figure(3, figsize=(8, 6))
    color_list = [(200/255, 158/255, 116/255), (143/255, 187/255, 218/255), (150/255, 208/255, 150/255), (153/255, 135/255, 206/255), (248/255,149/255,136/255)]
    ax1 = plt.subplot(111)
    input_mttc = real_world_scenarios.fusion_mttc
    sample_mttc = derived_scenarios.fusion_mttc
    sample_2_mttc = derived_2_scenarios.fusion_mttc
    sample_3_mttc = derived_3_scenarios.fusion_mttc
    sample_4_mttc = derived_4_scenarios.fusion_mttc
    bin_edges = np.arange(0, 5, 0.3)

    sns.histplot(data=input_mttc, bins=bin_edges, label="Real-world Scenarios", ax=ax1, color=color_list[0], stat='probability', element='bars', linewidth=0.1)
    sns.histplot(data=sample_mttc, bins=bin_edges, label="Non-directional Derived Scenarios", ax=ax1, color=color_list[1], stat='probability', element='bars', linewidth=0.1)
    sns.histplot(data=sample_2_mttc, bins=bin_edges, label="Directional Derived Scenarios between 0-1", ax=ax1, color=color_list[2], stat='probability', element='bars', linewidth=0.1)
    # sns.histplot(data=sample_3_mttc, bins=bin_edges, label="Directional derived Scenarios between 1-2.5", ax=ax1, color=color_list[3], stat='probability', element='bars', linewidth=0.1)
    # sns.histplot(data=sample_4_mttc, bins=bin_edges, label="Directional derived Scenarios between 2.5-4", ax=ax1, color=color_list[4], stat='probability', element='bars', linewidth=0.1)
    ax1.legend(fontsize=17.5)
    ax1.set_ylabel('Density', fontsize=28)
    ax1.set_xlabel('Modified time to collision (MTTC) (s)', fontsize=27)
    ax1.set_xlim((-0.5, 5.5))
    ax1.set_ylim((0, 0.36))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.savefig('./processed_by_zwt/image/minMTTC_hist.svg', dpi=600)
    # 绘制THW分布
    # plt.clf()
    # ax2 = plt.subplot(122)
    # input_thw = real_world_scenarios.fusion_thw
    # sample_thw = derived_scenarios.fusion_thw
    # sns.histplot(data=input_thw, bins=bin_edges, label="real-world Scenarios", ax=ax2, color=color_list[0], stat='probability', element='bars', linewidth=0.1)
    # sns.histplot(data=sample_thw, bins=bin_edges, label="derived Scenarios", ax=ax2, color=color_list[1], stat='probability', element='bars', linewidth=0.1)
    # # sns.histplot(data=input_thw, bins=20, label="real-world Scenarios", ax=ax2, color=color_list[0], stat='density', element='bars')
    # # sns.histplot(data=sample_thw, bins=20, label="derived Scenarios", ax=ax2, color=color_list[1], stat='density', element='bars')
    # ax2.legend()
    # ax2.set_ylabel('Density')
    # ax2.set_xlabel('Minimal time headway (THW) (s)')
    # ax2.set_xlim((-0.5, 5.5))
    plt.tight_layout()
    plt.savefig('./processed_by_zwt/image/minMTTC_compare_hist_0.3.svg', dpi=600)
    plt.clf()
    return

def calc_kl_divergence(x, y, bin_edges):
    # 使用numpy.digitize函数找到每个元素所在的区间索引
    indices = np.digitize(x, bin_edges) - 1

    # 将列表中的元素替换为所在区间的下边界
    replaced_x = list(bin_edges[indices])

    # 使用numpy.digitize函数找到每个元素所在的区间索引
    indices = np.digitize(y, bin_edges) - 1

    # 将列表中的元素替换为所在区间的下边界
    replaced_y = list(bin_edges[indices])

    k_x = set(replaced_x)
    p = []
    for i in k_x:
        p.append(replaced_x.count(i) / len(replaced_x))

    k_y = set(replaced_y)
    q = []
    for i in k_y:
        q.append(replaced_y.count(i) / len(replaced_y))

    KL = 0.0
    for i in range(len(k_x)):
        KL += p[i] * log(p[i] / q[i], 2)
    print(round(KL, 2))

def calc_kl_divergence_version_2(x, y, bin_edges):
    x_hist_counts, _ = np.histogram(x, bins=bin_edges)
    y_hist_counts, _ = np.histogram(y, bins=bin_edges)
    x_hist_counts = x_hist_counts / len(x)
    y_hist_counts = y_hist_counts / len(y)
    KL = 0
    for i in range(len(x_hist_counts)):
        try:
            KL += x_hist_counts[i] * log(x_hist_counts[i] / y_hist_counts[i], 2)
        except ValueError:
            KL += x_hist_counts[i] * log(x_hist_counts[i] / (y_hist_counts[i] + 1e-8), 2)
    print(KL)

if __name__ == '__main__':
    x1_min, x1_max = 0.0, 235.5454259747434
    y1_min, y1_max = -3.679218247347115, 2.5978529242443447
    x2_min, x2_max = -19.260000000000048, 385.4497701569932
    y2_min, y2_max = -5.030000000000001, 6.909999999999997
    x3_min, x3_max = 8.329999999999998, 413.4503592184133
    y3_min, y3_max = -3.460082480103437, 3.69
    lims = [x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max, x3_min, x3_max, y3_min, y3_max]

    global_trainScene_path = r'./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train.npy'
    global_sampleScene_without_latent_path = r'./processed_by_zwt/samples/data_modify_once.npy'
    global_sampleScene_with_latent_path = r'./processed_by_zwt/samples/data_modify_twice.npy'
    global_sampleScene_with_latent_1_2_path = r'./processed_by_zwt/samples/data_modify_twice_1_2.5.npy'
    global_sampleScene_with_latent_2_4_path = r'./processed_by_zwt/samples/data_modify_twice_2.5_4.npy'

    global_trainScene = Repair_cutin(fusion_path=global_trainScene_path, lims=lims, isNorm=True, need_repair=False)
    global_sampleScene_without_latent = Repair_cutin(fusion_path=global_sampleScene_without_latent_path, lims=lims, isNorm=True, need_repair=False)
    global_sampleScene_with_latent = Repair_cutin(fusion_path=global_sampleScene_with_latent_path, lims=lims, isNorm=True, need_repair=False)
    global_sampleScene_with_latent_1_2 = Repair_cutin(fusion_path=global_sampleScene_with_latent_1_2_path, lims=lims, isNorm=True, need_repair=False)
    global_sampleScene_with_latent_2_4 = Repair_cutin(fusion_path=global_sampleScene_with_latent_2_4_path, lims=lims, isNorm=True, need_repair=False)

    print(len(global_trainScene.fusion_mttc))  # 3186
    print(len(global_sampleScene_with_latent.fusion_mttc))  # 3100
    print(len(global_trainScene.fusion_thw))  # 3186
    print(len(global_sampleScene_with_latent.fusion_thw))  # 3100
    global_sampleScene_without_latent.fusion_mttc = [i for i in global_sampleScene_without_latent.fusion_mttc if i < 5.5]  # 2731
    global_sampleScene_without_latent.fusion_thw = [i for i in global_sampleScene_without_latent.fusion_thw if i < 5.5]  # 3029
    global_sampleScene_with_latent.fusion_mttc = [i for i in global_sampleScene_with_latent.fusion_mttc if i < 5.5]  # 2731
    global_sampleScene_with_latent.fusion_thw = [i for i in global_sampleScene_with_latent.fusion_thw if i < 5.5]  # 3029
    global_sampleScene_with_latent_1_2.fusion_mttc = [i for i in global_sampleScene_with_latent_1_2.fusion_mttc if i < 5.5]
    global_sampleScene_with_latent_1_2.fusion_thw = [i for i in global_sampleScene_with_latent_1_2.fusion_thw if i < 5.5]
    global_sampleScene_with_latent_2_4.fusion_mttc = [i for i in global_sampleScene_with_latent_2_4.fusion_mttc if i < 5.5]
    global_sampleScene_with_latent_2_4.fusion_thw = [i for i in global_sampleScene_with_latent_2_4.fusion_thw if i < 5.5]

    # 画图
    # plot_distribution(global_trainScene, global_sampleScene_with_latent)
    plot_hist(global_trainScene, global_sampleScene_without_latent, global_sampleScene_with_latent, global_sampleScene_with_latent_1_2, global_sampleScene_with_latent_2_4)

    # 计算KL散度(这里有部分场景的mttc被去掉了)
    # bin_edges = np.arange(0, 5, 0.3)
    # calc_kl_divergence_version_2(global_trainScene.fusion_mttc, global_sampleScene_with_latent.fusion_mttc, bin_edges)
    # calc_kl_divergence_version_2(global_trainScene.fusion_thw, global_sampleScene_with_latent.fusion_thw, bin_edges)


