import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


def restore(pos, min_p, max_p):
    """将归一化的参数还原
    :param pos        : 某一维参数序列对应的列表
    :param min_p      : 该参数归一化时对应的下界
    :param max_p      : 该参数归一化对应的上界

    :return pos       : 完成还原的参数序列
    """
    pos = [min_p + i * (max_p - min_p) for i in pos]
    return pos


def get_speed_and_accer(position_series, rate):
    """
    通过xy差分得到速度，加速度和jerk
    xy -> v -> a -> jerk
    """
    x1 = position_series[:, 0].tolist()
    y1 = position_series[:, 1].tolist()
    x2 = position_series[:, 2].tolist()
    y2 = position_series[:, 3].tolist()
    x3 = position_series[:, 4].tolist()
    y3 = position_series[:, 5].tolist()
    positions = np.array([x1, y1, x2, y2, x3, y3]).astype(float)
    velocity = np.array([[0] * len(x1) for _ in range(6)]).astype(float)
    accer = np.array([[0] * len(x1) for _ in range(6)]).astype(float)
    jerk = np.array([[0] * len(x1) for _ in range(6)]).astype(float)
    # 差分计算速度（由前后两个点差分得到——中心差分）
    for i in range(0, len(x1)):
        if i == 0:
            for j in range(6):
                velocity[j, i] = (positions[j, i + 2] - positions[j, i]) / float(2 * rate)
        elif i == len(x1) - 1:
            for j in range(6):
                velocity[j, i] = (positions[j, i] - positions[j, i - 2]) / float(2 * rate)
        else:
            for j in range(6):
                velocity[j, i] = (positions[j, i + 1] - positions[j, i - 1]) / float(2 * rate)

    # 差分计算加速度（由前后两个点差分得到）
    for i in range(0, len(x1)):
        if i == 0:
            for j in range(6):
                accer[j, i] = (velocity[j, i + 2] - velocity[j, i]) / float(2 * rate)
        elif i == len(x1) - 1:
            for j in range(6):
                accer[j, i] = (velocity[j, i] - velocity[j, i - 2]) / float(2 * rate)
        else:
            for j in range(6):
                accer[j, i] = (velocity[j, i + 1] - velocity[j, i - 1]) / float(2 * rate)
    # 差分计算加加速度（中心差分）
    for i in range(0, len(x1)):
        if i == 0:
            for j in range(6):
                jerk[j, i] = (accer[j, i + 2] - accer[j, i]) / float(2 * rate)
        elif i == len(x1) - 1:
            for j in range(6):
                jerk[j, i] = (accer[j, i] - accer[j, i - 2]) / float(2 * rate)
        else:
            for j in range(6):
                jerk[j, i] = (accer[j, i + 1] - accer[j, i - 1]) / float(2 * rate)
    # v, a = np.array(velocity), np.array(accer)
    return velocity, accer, jerk


def getL2(pos1: list, pos2: list):
    """计算两点之间的L2距离
    """
    disL2 = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return disL2


def jerkAnalysis(jerk):
    """离群点分析——极端突变值比例与符号反转频率
    :param jerk          : 传入的加加速度张量[nums_scene, length_scene]

    :return jerk_single  : 平均每个场景包含的极端突变值比例
    """
    jerk_table = [0] * jerk.shape[0]  # 数组哈希表，每个下标对应一个场景，元素值代表该场景中极端突变值的帧数
    index_ = np.where(jerk > 15)
    jerk_ind = np.concatenate((index_[0].reshape([-1, 1]), index_[1].reshape([-1, 1])), axis=1)  # 将离群点坐标合并[nums_outliar, 2]——2代表离群点位置[scene_id, frame_id]
    for i in jerk_ind:
        jerk_table[i[0]] += 1
    jerk_single = 0  # 平均每个场景的极端突变值比例
    if sum(jerk_table):
        jerk_single = np.mean([round(i / jerk.shape[1], 4) for i in jerk_table])
    return jerk_single


def wavefilter(input):
    '''小波滤波——多贝西6，高频全置零
    :param input      : 待滤波的所有场景某一维参数的集合[nums_scenes, length_scenes, 1]

    :return output    : 完成滤波的所有场景某一维参数的集合[nums_scenes, length_scenes]
    '''
    output = np.array([])
    for i in range(input.shape[0]):
        data = input[i].reshape(-1)
        # We will use the Daubechies(6) wavelet
        daubechies_num = 6
        wname = "db" + str(daubechies_num)
        datalength = data.shape[0]
        wavelet = pywt.Wavelet(wname)
        max_level = pywt.dwt_max_level(datalength, wavelet.dec_len)
        # print('maximun level is: %s' % max_level)
        # Initialize the container for the filtered data
        # Decompose the signal
        # coeff[0] is approximation coeffs, coeffs[1] is nth level detail coeff, coeff[-1] is first level detail coeffs
        coeffs = pywt.wavedec(data, wname, mode='smooth', level=max_level)
        # thresholding
        for j in range(max_level):
            coeffs[-j - 1] = np.zeros_like(coeffs[-j - 1])
        # Reconstruct the signal and save it
        filter_data = pywt.waverec(coeffs, wname, mode='smooth')
        fdata = filter_data[0:datalength]
        if output.size > 0:
            output = np.concatenate((output, fdata.reshape(1, -1)), axis=0)
        else:
            output = fdata.reshape(1, -1).copy()
    return output


def outliarRemoval(a, v, a_bound, flag=True):
    """剔除离群点——在加速度曲线上识别，在速度曲线上修正
    :param a          : 传入的加速度张量[nums_scene, length_scene]
    :param v          : 传入的速度张量[nums_scene, length_scene]
    :param a_bound    : 加速度边界
    :param flag       : 是否直接在传入的速度张量上做离群点剔除

    :return v         : 完成离群点剔除的速度张量[nums_scene, length_scene]
    :return RR_single : 平均每个场景需要被修复的帧数比例
    :return RR_all    : 需要被修复的场景比例
    """
    repair_table = [0] * a.shape[0]  # 数组哈希表，每个下标对应一个场景，元素值代表该场景中需要修复的帧数
    index_ = np.where((a < a_bound[0]) | (a > a_bound[1]))
    repair_ind = np.concatenate((index_[0].reshape([-1, 1]), index_[1].reshape([-1, 1])), axis=1)  # 将离群点坐标合并[nums_outliar, 2]——2代表离群点位置[scene_id, frame_id]
    for i in repair_ind:
        if i[1] - 1 >= 0 and i[1] + 1 < a.shape[1]:
            if flag:
                pre_status = v[i[0], i[1] - 1].reshape([-1])
                next_status = v[i[0], i[1] + 1].reshape([-1])
                v_new = np.mean(np.concatenate((pre_status, next_status)))  # 离群点使用其前后状态的均值做替代
                v[i[0], i[1]] = v_new
            repair_table[i[0]] += 1
    RR_single, RR_all = 0, 0
    if sum(repair_table):
        RR_single = np.mean([round(i / a.shape[1], 4) for i in repair_table])  # 每个场景平均需要被修复的帧数比例
    RR_all = round(len(np.unique(index_[0])) / a.shape[0], 4)  # 需要被修复的场景数比例
    return v, RR_single, RR_all


def cutinPloter(inputScene_xy):
    '''cutin场景可视化'''
    for ind in range(inputScene_xy.shape[0]):
        ego_x = inputScene_xy[ind, :, 0]
        ego_y = inputScene_xy[ind, :, 1]
        cutin_x = inputScene_xy[ind, :, 2]
        cutin_y = inputScene_xy[ind, :, 3]

        min_x = min(min(ego_x), min(cutin_x)) - 10
        max_x = max(max(ego_x), max(cutin_x)) + 10
        min_y = min(min(ego_y), min(cutin_y)) - 10
        max_y = max(max(ego_y), max(cutin_y)) + 10

        plt.ion()
        plt.subplot()
        for i in range(len(ego_x)):
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
            plt.scatter(ego_x[i], ego_y[i], c='r')
            plt.scatter(cutin_x[i], cutin_y[i], c='b')
            plt.annotate('主车', xy=(ego_x[i], ego_y[i]), xytext=(-10, 6), textcoords='offset points')
            plt.annotate('对手车', xy=(cutin_x[i], cutin_y[i]), xytext=(-10, -13), textcoords='offset points')
            plt.pause(1e-7)
            plt.clf()
        plt.ioff()
    return
