
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
import math
from subFunctions import *
from data_procession import *
import warnings
warnings.filterwarnings('ignore')


class Repair_cutin():
    def __init__(self, frame_rate=0.04, lon_bound=[-8, 5], lat_bound=[-3.5, 3.5], fusion_path='', isNorm=True, isXy=True, lims=[], need_repair=True) -> None:
        self.rate = frame_rate
        self.isNorm = isNorm
        self.isXy = isXy
        self.lims = lims
        self.need_repair = need_repair
        self.lon_bound = lon_bound
        self.lat_bound = lat_bound
        self.fusion = np.load(fusion_path, allow_pickle=True)
        self.fusion_re = np.array([])
        self.fusion_xy = np.array([])
        self.fusion_v = np.array([])
        self.fusion_a = np.array([])
        self.fusion_j = np.array([])
        self.realFlag = [0] * self.fusion.shape[0]

        self.fusion_v_outliar = np.array([])
        self.fusion_v_denosie = np.array([])
        self.fusion_xy_repair = np.array([])
        self.fusion_v_repair = np.array([])
        self.fusion_a_repair = np.array([])
        self.fusion_j_repair = np.array([])
        self.fusion_mttc = []
        self.fusion_thw = []
        self.sceneRecoverPosition()
        self.sceneEnhace()
        self.isCutin_by_zwt()
        self.getMttc()
        print('over')

    def sceneRecoverPosition(self):
        print(self.fusion.shape[0])
        for i in range(self.fusion.shape[0]):
            x1 = self.fusion[i, :, 0].tolist()
            y1 = self.fusion[i, :, 1].tolist()
            x2 = self.fusion[i, :, 2].tolist()
            y2 = self.fusion[i, :, 3].tolist()
            x3 = self.fusion[i, :, 4].tolist()
            y3 = self.fusion[i, :, 5].tolist()

            if self.isNorm:
                x1 = restore(x1, self.lims[0], self.lims[1])
                y1 = restore(y1, self.lims[2], self.lims[3])
                x2 = restore(x2, self.lims[4], self.lims[5])
                y2 = restore(y2, self.lims[6], self.lims[7])
                x3 = restore(x3, self.lims[8], self.lims[9])
                y3 = restore(y3, self.lims[10], self.lims[11])
            temp_array1 = np.empty((len(x1), 6), dtype=np.float64)
            temp_array1[:, 0] = x1
            temp_array1[:, 1] = y1
            temp_array1[:, 2] = x2
            temp_array1[:, 3] = y2
            temp_array1[:, 4] = x3
            temp_array1[:, 5] = y3
            if self.fusion_re.size > 0:
                self.fusion_re = np.concatenate((self.fusion_re, temp_array1.reshape([1, -1, 6])), axis=0)
            else:
                self.fusion_re = temp_array1.reshape([1, -1, 6])

            velocity, accer, jerk = get_speed_and_accer(temp_array1, self.rate)
            if self.fusion_v.size > 0:
                self.fusion_v = np.concatenate((self.fusion_v, velocity.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_a = np.concatenate((self.fusion_a, accer.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_j = np.concatenate((self.fusion_j, jerk.transpose().reshape([1, -1, 6])), axis=0)
            else:
                self.fusion_v = velocity.transpose().reshape([1, -1, 6])
                self.fusion_a = accer.transpose().reshape([1, -1, 6])
                self.fusion_j = jerk.transpose().reshape([1, -1, 6])

        self.fusion_xy = self.fusion_re.copy()
        return

    def sceneEnhace(self):
        if not self.need_repair:
            self.fusion_xy_repair = self.fusion_xy.copy()
            self.fusion_v_repair = self.fusion_v.copy()
            self.fusion_v_denosie = self.fusion_v.copy()
            self.fusion_a_repair = self.fusion_a.copy()
            self.fusion_j_repair = self.fusion_j.copy()
            print("跳过数据强化环节")
            np.save('./processed/output/cutin3.npy', self.fusion_xy)
            print("Saved the sample scenarios")

            ego_x_jerk = jerkAnalysis(self.fusion_j[:, :, 0])
            ego_y_jerk = jerkAnalysis(self.fusion_j[:, :, 1])
            cutin_x_jerk = jerkAnalysis(self.fusion_j[:, :, 2])
            cutin_y_jerk = jerkAnalysis(self.fusion_j[:, :, 3])
            pre_x_jerk = jerkAnalysis(self.fusion_j[:, :, 4])
            pre_y_jerk = jerkAnalysis(self.fusion_j[:, :, 5])
            print('增强前的极端突变值比例：{},{},{},{},{},{}'.format(ego_x_jerk, ego_y_jerk, cutin_x_jerk, cutin_y_jerk, pre_x_jerk, pre_y_jerk))

            self.fusion_v_outliar = self.fusion_v.copy()
            ego_x_outliar = outliarRemoval(self.fusion_a[:, :, 0], self.fusion_v_outliar[:, :, 0], self.lon_bound)
            ego_y_outliar = outliarRemoval(self.fusion_a[:, :, 1], self.fusion_v_outliar[:, :, 1], self.lat_bound)
            cutin_x_outliar = outliarRemoval(self.fusion_a[:, :, 2], self.fusion_v_outliar[:, :, 2], self.lon_bound)
            cutin_y_outliar = outliarRemoval(self.fusion_a[:, :, 3], self.fusion_v_outliar[:, :, 3], self.lat_bound)
            pre_x_outliar = outliarRemoval(self.fusion_a[:, :, 4], self.fusion_v_outliar[:, :, 4], self.lon_bound)
            pre_y_outliar = outliarRemoval(self.fusion_a[:, :, 5], self.fusion_v_outliar[:, :, 5], self.lat_bound)
            self.fusion_v_outliar = np.dstack((ego_x_outliar[0], ego_y_outliar[0], cutin_x_outliar[0], cutin_y_outliar[0], pre_x_outliar[0], pre_y_outliar[0]))
            print('增强前的动力学检查结果：{},{},{},{},{},{}'.format(ego_x_outliar[1:], ego_y_outliar[1:],
                  cutin_x_outliar[1:], cutin_y_outliar[1:], pre_x_outliar[1:], pre_y_outliar[1:]))
            return

        noSolution_count = 0
        all_outlier_record = pd.DataFrame()
        for scene_ind in tqdm(range(self.fusion_xy.shape[0])):
            ego_x = self.fusion_xy[scene_ind, :, 0]
            ego_y = self.fusion_xy[scene_ind, :, 1]
            ego_vx = self.fusion_v[scene_ind, :, 0]
            ego_vy = self.fusion_v[scene_ind, :, 1]
            for i in range(2):
                cutin_x = self.fusion_xy[scene_ind, :, 2 * (i + 1)]
                cutin_y = self.fusion_xy[scene_ind, :, 2 * (i + 1) + 1]
                cutin_vx = self.fusion_v[scene_ind, :, 2 * (i + 1)]
                cutin_vy = self.fusion_v[scene_ind, :, 2 * (i + 1) + 1]
                df_ego, df_cutin = pd.DataFrame(), pd.DataFrame()
                df_ego['local_time_stamp'] = np.arange(1, 126, 1)
                df_ego['segment_id'] = scene_ind * np.ones(125)
                df_ego['veh_id'] = np.zeros(125)
                df_ego['length'] = 4 * np.ones(125)
                df_ego['global_center_x'] = ego_x
                df_ego['global_center_y'] = ego_y
                df_ego['speed_x'] = ego_vx
                df_ego['speed_y'] = ego_vy
                df_cutin['local_time_stamp'] = np.arange(1, 126, 1)
                df_cutin['segment_id'] = scene_ind * np.ones(125)
                df_cutin['veh_id'] = np.ones(125)
                df_cutin['length'] = 4 * np.ones(125)
                df_cutin['global_center_x'] = cutin_x
                df_cutin['global_center_y'] = cutin_y
                df_cutin['speed_x'] = cutin_vx
                df_cutin['speed_y'] = cutin_vy
                if df_ego.loc[0, 'global_center_x'] > df_ego.loc[len(df_ego) - 1, 'global_center_x']:
                    df_ego.loc[:, 'global_center_x'] = df_ego.loc[:, 'global_center_x'] * -1
                    df_ego.loc[:, 'global_center_y'] = df_ego.loc[:, 'global_center_y'] * -1
                    df_ego.loc[:, 'speed_x'] = df_ego.loc[:, 'speed_x'] * -1
                    df_ego.loc[:, 'speed_y'] = df_ego.loc[:, 'speed_y'] * -1
                    df_cutin.loc[:, 'global_center_x'] = df_cutin.loc[:, 'global_center_x'] * -1
                    df_cutin.loc[:, 'global_center_y'] = df_cutin.loc[:, 'global_center_y'] * -1
                    df_cutin.loc[:, 'speed_x'] = df_cutin.loc[:, 'speed_x'] * -1
                    df_cutin.loc[:, 'speed_y'] = df_cutin.loc[:, 'speed_y'] * -1

                out_trj = pd.DataFrame()
                out_trj = pair_cf_coord_cal(0, df_ego, 1, df_cutin, 0, all_outlier_record)
                if not out_trj.empty:
                    df_final = out_trj.loc[:, ['local_veh_id', 'length', 'local_time', 'filter_pos_x', 'filter_speed_x',
                                           'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']]
                    ego, cutin = df_final[df_final['local_veh_id'] == 0], df_final[df_final['local_veh_id'] == 1]
                    ego_x_repair = np.array(ego['filter_pos_x'])
                    ego_y_repair = np.array(ego['filter_pos_y'])
                    ego_vx_repair = np.array(ego['filter_speed_x'])
                    ego_vy_repair = np.array(ego['filter_speed_y'])
                    ego_ax_repair = np.array(ego['filter_accer_x'])
                    ego_ay_repair = np.array(ego['filter_accer_y'])
                    cutin_x_repair = np.array(cutin['filter_pos_x'])
                    cutin_y_repair = np.array(cutin['filter_pos_y'])
                    cutin_vx_repair = np.array(cutin['filter_speed_x'])
                    cutin_vy_repair = np.array(cutin['filter_speed_y'])
                    cutin_ax_repair = np.array(cutin['filter_accer_x'])
                    cutin_ay_repair = np.array(cutin['filter_accer_y'])
                    if i == 0:
                        temp_array_xy = np.dstack((ego_x_repair, ego_y_repair, cutin_x_repair, cutin_y_repair))
                        temp_array_v = np.dstack((ego_vx_repair, ego_vy_repair, cutin_vx_repair, cutin_vy_repair))
                        temp_array_a = np.dstack((ego_ax_repair, ego_ay_repair, cutin_ax_repair, cutin_ay_repair))
                    else:
                        temp_array_xy = np.dstack((temp_array_xy, cutin_x_repair, cutin_y_repair))
                        temp_array_v = np.dstack((temp_array_v, cutin_vx_repair, cutin_vy_repair))
                        temp_array_a = np.dstack((temp_array_a, cutin_ax_repair, cutin_ay_repair))
                        if self.fusion_xy_repair.size > 0:
                            self.fusion_xy_repair = np.concatenate((self.fusion_xy_repair, temp_array_xy.reshape([1, -1, 6])), axis=0)
                            self.fusion_v_repair = np.concatenate((self.fusion_v_repair, temp_array_v.reshape([1, -1, 6])), axis=0)
                            self.fusion_a_repair = np.concatenate((self.fusion_a_repair, temp_array_a.reshape([1, -1, 6])), axis=0)
                        else:
                            self.fusion_xy_repair = temp_array_xy.reshape([1, -1, 6])
                            self.fusion_v_repair = temp_array_v.reshape([1, -1, 6])
                            self.fusion_a_repair = temp_array_a.reshape([1, -1, 6])
                else:
                    noSolution_count += 1
                    break
        print('无法通过数据增强实现场景修复的场景比例：{}%'.format(100 * (noSolution_count / self.fusion_xy.shape[0])))
        np.save('./processed/output/cutin3_processed.npy', self.fusion_xy_repair)
        print("Saved the processed sample scenarios")
        self.fusion_v_denosie = self.fusion_v.copy()
        return

    def reconEval(self, orig_input):
        evalADE, evalFDE = [], []
        for i in range(self.fusion.shape[0]):
            ade, fde = 0, 0
            for j in range(self.fusion.shape[1]):
                ade += (getL2(self.fusion_xy[i, j, :2], orig_input[i, j, :2]) +
                        getL2(self.fusion_xy[i, j, 2:4], orig_input[i, j, 2:4]) +
                        getL2(self.fusion_xy[i, j, 4:], orig_input[i, j, 4:])) / 3
                if j == self.fusion.shape[1] - 1:
                    fde = (getL2(self.fusion_xy[i, j, :2], orig_input[i, j, :2]) +
                           getL2(self.fusion_xy[i, j, 2:4], orig_input[i, j, 2:4]) +
                           getL2(self.fusion_xy[i, j, 4:], orig_input[i, j, 4:])) / 3
            evalADE.append(ade / self.fusion.shape[1])
            evalFDE.append(fde)
        return sum(evalADE) / len(evalADE), sum(evalFDE) / len(evalFDE)

    def isCutin(self):
        v_ego = self.fusion_v_repair[:, :, 0]
        if not self.isXy:
            delta_y = self.fusion_re[:, :, 1]
            delta_x = self.fusion_re[:, :, 0]
        else:
            delta_y = self.fusion_xy_repair[:, :, 1] - self.fusion_xy_repair[:, :, 3]
            delta_x = self.fusion_xy_repair[:, :, 0] - self.fusion_xy_repair[:, :, 2]
        vaild_count = 0
        c1, c2, c3 = 0, 0, 0
        for i in range(delta_y.shape[0]):
            y = list(delta_y[i, :])
            v = list(v_ego[i, :])
            x = list(delta_x[i, :])
            flag1, flag2, flag3 = True, True, False
            # C1
            if abs(y[-1]) > 1.5:
                flag1 = False
            else:
                c1 += 1
            # C2
            if abs(y[0]) < 1.5:
                flag2 = False
            else:
                c2 += 1
            # C3
            for j in range(len(y) - 1, -1, -1):
                if abs(y[j]) < 1.5 and abs(x[j]) / v[j] < 5:
                    flag3 = True
                    c3 += 1
                    break
            if flag1 and flag2 and flag3:
                vaild_count += 1
                self.realFlag[i] = 1
        result = [vaild_count / delta_y.shape[0], c1 / delta_y.shape[0], c2 / delta_y.shape[0], c3 / delta_y.shape[0]]
        print('C: {0[0]}, C1: {0[1]}, C2: {0[2]}, C3: {0[3]}'.format(result))
        return

    def isCutin_by_zwt(self):
        v_ego = self.fusion_v_denosie[:, :, 0]
        if not self.isXy:
            delta_y = self.fusion_re[:, :, 1]
            delta_x = self.fusion_re[:, :, 0]
        else:
            delta_y = self.fusion_xy_repair[:, :, 1] - self.fusion_xy_repair[:, :, 3]
            delta_x = self.fusion_xy_repair[:, :, 0] - self.fusion_xy_repair[:, :, 2]
        vaild_count = 0
        c1, c2, c3 = 0, 0, 0
        for i in range(delta_y.shape[0]):
            y = list(delta_y[i, :])
            v = list(v_ego[i, :])
            x = list(delta_x[i, :])
            flag1, flag2, flag3 = True, True, True
            for j in range(len(y) - 1, len(y) - 26, -1):
                if abs(y[j]) > 1.5:
                    flag1 = False
                    break
            if flag1:
                c1 += 1
            if abs(y[0]) < 1.5:
                flag2 = False
            else:
                c2 += 1
            temp_ids = None
            for j in range(0, len(y) - 1):
                if abs(y[j]) < 1.5:
                    temp_ids = j
                    break
            if temp_ids is not None:
                for k in range(temp_ids, len(y) - 1):
                    if abs(x[k]) / v[k] > 5:
                        flag3 = False
                        break
            else:
                flag3 = False
            if flag3:
                c3 += 1
            if flag1 and flag2 and flag3:
                vaild_count += 1
                self.realFlag[i] = 1
        result = [vaild_count / delta_y.shape[0], c1 / delta_y.shape[0], c2 / delta_y.shape[0], c3 / delta_y.shape[0]]
        print('C: {0[0]}, C1: {0[1]}, C2: {0[2]}, C3: {0[3]}'.format(result))
        return

    def isReal(self):
        ego_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 0], self.fusion_v_repair[:, :, 0], self.lon_bound, flag=False)
        ego_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 1], self.fusion_v_repair[:, :, 1], self.lat_bound, flag=False)
        cutin_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 2], self.fusion_v_repair[:, :, 2], self.lon_bound, flag=False)
        cutin_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 3], self.fusion_v_repair[:, :, 3], self.lat_bound, flag=False)
        pre_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 4], self.fusion_v_repair[:, :, 4], self.lon_bound, flag=False)
        pre_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 5], self.fusion_v_repair[:, :, 5], self.lat_bound, flag=False)
        print('增强后的动力学检查结果：{},{},{},{},{},{}'.format(ego_x_outliar[1:], ego_y_outliar[1:],
              cutin_x_outliar[1:], cutin_y_outliar[1:], pre_x_outliar[1:], pre_y_outliar[1:]))

        ego_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 0])
        ego_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 1])
        cutin_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 2])
        cutin_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 3])
        pre_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 4])
        pre_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 5])
        print('增强后的极端突变值比例：{}, {}, {}, {}, {}, {}'.format(ego_x_jerk, ego_y_jerk, cutin_x_jerk, cutin_y_jerk, pre_x_jerk, pre_y_jerk))
        return

    def isVaild(self, test, test_xy):
        def getPattern(nums):
            '''将序列转换为模式序列'''
            nums = nums[:, ::3]
            diff_num = np.diff(nums, axis=1)
            pattern = np.where(diff_num > 0, 1, diff_num)
            pattern = np.where(pattern < 0, -1, pattern)
            return pattern

        # denoise = self.wavefilter(self.fusion_re[:, :, 0])
        for i in range(self.fusion_v_repair.shape[0]):
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.ion()
            plt.subplot(2, 1, 1)
            plt.title('纵向速度（上）及横向速度（下）曲线')
            plt.plot(self.fusion_v_repair[i, :, 0])
            plt.ylim(-200, 0)
            plt.subplot(2, 1, 2)
            plt.plot(self.fusion_v_repair[i, :, 0])
            plt.ylim(-200, 0)
            plt.pause(2)
            plt.clf()
            plt.ioff()
        pattern_dis_ego = np.mean(np.abs(pattern_ego - pattern_true_ego), axis=1)
        ego = []
        for i in range(v_ego.shape[0]):
            r, p = stats.pearsonr(pattern_true_ego[i, :], pattern_ego[i, :])
            ego.append([r, p])
        pattern_dis_cutin = []
        pearson_cutin = []
        for i in range(2, 4):
            pattern_true = getPattern(test[:, :, i])
            pattern_fusion = getPattern(self.fusion[:, :, i])
            pattern_dis_cutin.append(np.mean(np.abs(pattern_fusion - pattern_true), axis=1))
            pearson_cutin.append([stats.pearsonr(pattern_true[i, :], pattern_fusion[i, :]) for i in range(test.shape[0])])
        print(np.mean(pattern_dis_ego), np.mean(pattern_dis_cutin[0]), np.mean(pattern_dis_cutin[1]))
        return
    
    def getMttc(self):
        for i in range(self.fusion_xy_repair.shape[0]):
            if not self.realFlag[i]:
                continue
            cur_mttc = []
            cur_thw = []
            for j in range(self.fusion_xy_repair.shape[1] - 1, -1, -1):
                if abs(self.fusion_xy_repair[i, j, 1] - self.fusion_xy_repair[i, j, 3]) <= 1.5:
                    dist = abs(self.fusion_xy_repair[i, j, 0] - self.fusion_xy_repair[i, j, 2])
                    # delta_v = abs(self.fusion_v_repair[i, j, 0]) - abs(self.fusion_v_repair[i, j, 2])
                    delta_v = self.fusion_v_repair[i, j, 0] - self.fusion_v_repair[i, j, 2]
                    delta_v2 = (self.fusion_v_repair[i, j, 0] - self.fusion_v_repair[i, j, 2]) ** 2
                    delta_a = self.fusion_a_repair[i, j, 0] - self.fusion_a_repair[i, j, 2]
                    if delta_a != 0:
                        t1 = (delta_v * (-1) - (delta_v2 + 2 * delta_a * dist)**0.5) / delta_a
                        t2 = (delta_v * (-1) + (delta_v2 + 2 * delta_a * dist)**0.5) / delta_a
                        if t1 > 0 and t2 > 0:
                            cur_mttc.append(min(t1, t2))
                        elif t1 * t2 <= 0 and max(t1, t2) > 0:
                            cur_mttc.append(max(t1, t2))
                        else:
                            cur_mttc.append(dist / abs(self.fusion_v_repair[i, j, 0]))
                    cur_thw.append(dist / abs(self.fusion_v_repair[i, j, 0]))
                else:
                    # cur_mttc.append(float('inf'))
                    # cur_thw.append(float('inf'))
                    break
            try:
                self.fusion_mttc.append(min(cur_mttc))
            except ValueError:
                continue
            self.fusion_thw.append(min(cur_thw))
        return


if __name__ == '__main__':
    x1_min, x1_max = 0.0, 235.5454259747434
    y1_min, y1_max = -3.679218247347115, 2.5978529242443447
    x2_min, x2_max = -19.260000000000048, 385.4497701569932
    y2_min, y2_max = -5.030000000000001, 6.909999999999997
    x3_min, x3_max = 8.329999999999998, 413.4503592184133
    y3_min, y3_max = -3.460082480103437, 3.69
    lims = [x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max, x3_min, x3_max, y3_min, y3_max]
    threshold = [5, -8, 5, -8, 3.9, -3.9]
    sampleScene_orig = Repair_cutin(fusion_path='./E_WGAN/output/cutin3.npy', isNorm=False, need_repair=False)
    sampleScene_processed = Repair_cutin(fusion_path='./E_WGAN/output/cutin3_processed.npy', isNorm=False, need_repair=False)

    for i in range(sampleScene_orig.fusion_v_repair.shape[0]):
        time = np.linspace(0, 124, 125)
        correctness_dir = ['_x', '_y']
        for j in range(2):
            plt.subplot(3, 1, 1)
            plt.plot(time, sampleScene_orig.fusion_xy[i, :, j], '--k', linewidth=0.25, label='Original')
            plt.plot(time, sampleScene_processed.fusion_xy_repair[i, :, j], '-*g', linewidth=0.25, label='Processed', markersize=0.5)
            plt.ylabel('Position (m)')
            plt.legend(prop={'size': 6})
            trj_title = 'Scenario ' + str(i) + ' Ego' + ' Direction' + correctness_dir[j] + ' Before and After Filtering'
            plt.title(trj_title)
            plt.subplot(3, 1, 2)
            plt.plot(time, sampleScene_orig.fusion_v[i, :, j], '--k', linewidth=0.25, label='Original')
            plt.plot(time, sampleScene_processed.fusion_v_repair[i, :, j], '-*g', linewidth=0.25, label='Processed', markersize=0.5)
            plt.ylabel('Speed (m/s)')
            plt.legend(prop={'size': 6})
            if not j:
                plt.ylim([0, 50])
            else:
                plt.ylim([-10, 10])
            plt.subplot(3, 1, 3)
            plt.plot(time, sampleScene_orig.fusion_a[i, :, j], '--k', linewidth=0.25, label='Original')
            plt.plot(time, sampleScene_processed.fusion_a_repair[i, :, j], '-*g', linewidth=0.25, label='Processed', markersize=0.5)
            plt.legend(prop={'size': 6})
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s2)')
            if not j:
                plt.ylim([-15, 15])
            else:
                plt.ylim([-5, 5])
            if not os.path.exists('./VAE_WGAN/image/trajectory_process'):
                os.makedirs('./VAE_WGAN/image/trajectory_process')
            trj_save_title = './VAE_WGAN/image/trajectory_process/' + trj_title + '.png'
            plt.savefig(trj_save_title, dpi=600)
            plt.close('all')
        if i == 10:
            break

    # plt.show()

    # 场景可视化
    # cutinPloter(fusionScene.fusion_xy_repair)
