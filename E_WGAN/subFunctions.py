import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


def restore(pos, min_p, max_p):

    pos = [min_p + i * (max_p - min_p) for i in pos]
    return pos


def get_speed_and_accer(position_series, rate):
    """
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
    disL2 = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return disL2


def jerkAnalysis(jerk):
    jerk_table = [0] * jerk.shape[0]
    index_ = np.where(jerk > 15)
    jerk_ind = np.concatenate((index_[0].reshape([-1, 1]), index_[1].reshape([-1, 1])), axis=1)
    for i in jerk_ind:
        jerk_table[i[0]] += 1
    jerk_single = 0
    if sum(jerk_table):
        jerk_single = np.mean([round(i / jerk.shape[1], 4) for i in jerk_table])
    return jerk_single


def wavefilter(input):
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
    repair_table = [0] * a.shape[0]
    index_ = np.where((a < a_bound[0]) | (a > a_bound[1]))
    repair_ind = np.concatenate((index_[0].reshape([-1, 1]), index_[1].reshape([-1, 1])), axis=1)
    for i in repair_ind:
        if i[1] - 1 >= 0 and i[1] + 1 < a.shape[1]:
            if flag:
                pre_status = v[i[0], i[1] - 1].reshape([-1])
                next_status = v[i[0], i[1] + 1].reshape([-1])
                v_new = np.mean(np.concatenate((pre_status, next_status)))
                v[i[0], i[1]] = v_new
            repair_table[i[0]] += 1
    RR_single, RR_all = 0, 0
    if sum(repair_table):
        RR_single = np.mean([round(i / a.shape[1], 4) for i in repair_table])
    RR_all = round(len(np.unique(index_[0])) / a.shape[0], 4)
    return v, RR_single, RR_all


def cutinPloter(inputScene_xy):
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
            plt.rcParams['font.sans-serif'] = ['SimHei']
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
