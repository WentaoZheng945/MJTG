"""
@Date: 2023-02-17
特别注意：模型完成训练后进行评价时，出现cpu版本pytorch与gpu版结果差异巨大（cpu版全面碾压gpu版）
原因定位：与评价代码无关，仅仅是模型在不同版本pytorch下生成的场景质量存在差异
问题解决：在cutin3场景下，暂时无法解决，定位到cpu与gpu版torch在tensor.normal_()方法下输出结果差异巨大——model.py下的reparametrize()方法
权宜之计：在GPU下训练，在CPU下做泛化并评价
"""
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


# TODO: 对泛化的cutin场景进行后处理（数据增强），包括离群点去除及去噪
# 三车场景shape[num_scene, 125, 6]
# 其中六个维度分别为：主车xy坐标、执行cutin的对手车xy坐标（car2）、主车前车xy坐标（car3）
class Repair_cutin():
    def __init__(self, frame_rate=0.04, lon_bound=[-8, 5], lat_bound=[-3.5, 3.5], fusion_path='', isNorm=True, isXy=True, lims=[], need_repair=True) -> None:
        """场景修复所涉及到的场景采样率、加速度上下界及场景储存位置需要修改

        场景数据读入后将自动执行：1.归一化数据的还原；2. 反推主车、对手车的速度、加速度、加加速度；3. 数据增强（离群点剔除、滤波）；5. 功能性检查；4. 计算关键参数MTTC/THW

        :param frame_rate        : 采样率，每帧的时间间隔
        :param isNorm            : 标识传入数据是否为归一化的数据
        :param need_repair       : 标识传入数据是否需要数据增强
        :param lon_bound         : 场景纵向加速度的边界
        :param lat_bound         : 场景横向加速度的边界
        :param fusion_path       : 泛化场景的储存路径
        :param fusion            : 原始泛化场景数据[num_scene, len_scene, 6] -> [ego_x, ego_y, cutin_x, cutin_y, pre_x, pre_y] -> 顺序固定为ego/cutin/pre
        :param fusion_re         : 还原后的归一化泛化场景数据
        :param fusion_xy         : 坐标型泛化场景数据[num_scene, len_scene, 6]
        :param fusion_v          : 场景车辆速度数据[num_scene, len_scene, 6]
        :param fusion_a          : 场景车辆加速度数据[num_scene, len_scene, 6]
        :param realFlag          : 用于标识场景集合中各场景是否通过功能性检查[0, .., 0]→通过则置1
        :param fusion_j          : 场景车辆加加速度数据[num_scene, len_scene, 6]
        :param fusion_v_outliar  : 完成离群点去除的场景车辆速度数据[num_scene, len_scene, 6]
        :param fusion_v_denoise  : 完成去噪的场景车辆速度数据[num_scene, len_scene, 6]
        :param fusion_xy_repair  : 由完成数据增强的速度数据反推得到的场景xy坐标[num_scene, len_scene, 6]
        :param fusion_v_repair   : 由修复的场景xy数据差分得到的场景加速度数据[num_scene, len_scene, 6]
        :param fusion_a_repair   : 由修复的场景xy数据差分得到的场景加速度数据[num_scene, len_scene, 6]
        :param fusion_j_repair   : 由修复的场景xy数据差分得到的场景加加速度数据[num_scene, len_scene, 6]
        :param fusion_mttc       : 由修复的场景数据计算得到每个场景对应的最小MTTC
        :param fusion_thw        : 由修复的场景数据计算得到每个场景对应的最小THW
        """
        self.rate = frame_rate
        self.isNorm = isNorm
        self.isXy = isXy
        self.lims = lims  # 参数上下界，还原归一化使用 [x1_min, x1_max, y1_min, y1_max,...]->len:12
        self.need_repair = need_repair
        self.lon_bound = lon_bound
        self.lat_bound = lat_bound
        self.fusion = np.load(fusion_path, allow_pickle=True)
        self.fusion_re = np.array([])  # 归一化还原之后的xy数据
        self.fusion_xy = np.array([])  # fusion_re的深拷贝
        self.fusion_v = np.array([])  # fusion_xy差分得到
        self.fusion_a = np.array([])  # fusion_v差分得到
        self.fusion_j = np.array([])  # fusion_a差分得到
        self.realFlag = [0] * self.fusion.shape[0]

        self.fusion_v_outliar = np.array([])  # 去除离群点的速度数据
        self.fusion_v_denosie = np.array([])  # 去除噪声的速度数据
        self.fusion_xy_repair = np.array([])  # 由速度数据增强后的速度积分得到的xy
        self.fusion_v_repair = np.array([])  #
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
        """将传入的场景参数还原
        1. 将归一化的车辆坐标参数fusion，还原为原始量纲fusion_re
        2. 基于还原的fusion_re=fusion_xy，计算车辆速度加速度
        """
        print(self.fusion.shape[0])
        for i in range(self.fusion.shape[0]):
            x1 = self.fusion[i, :, 0].tolist()
            y1 = self.fusion[i, :, 1].tolist()
            x2 = self.fusion[i, :, 2].tolist()
            y2 = self.fusion[i, :, 3].tolist()
            x3 = self.fusion[i, :, 4].tolist()
            y3 = self.fusion[i, :, 5].tolist()

            # TODO: 将归一化的数据还原，并储存在fusion_re中
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

            # TODO: 反推基于位置的速度、加速度信息
            velocity, accer, jerk = get_speed_and_accer(temp_array1, self.rate)
            # 输出为对应单个场景[ego_x, ego_y, cutin_x, cutin_y, pre_x, pre_y], shape(6, 125)
            if self.fusion_v.size > 0:
                self.fusion_v = np.concatenate((self.fusion_v, velocity.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_a = np.concatenate((self.fusion_a, accer.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_j = np.concatenate((self.fusion_j, jerk.transpose().reshape([1, -1, 6])), axis=0)
            else:
                self.fusion_v = velocity.transpose().reshape([1, -1, 6])  # 对张量(6, 125)转置为(125, 6)，再reshape为(1, 125, 6)
                self.fusion_a = accer.transpose().reshape([1, -1, 6])
                self.fusion_j = jerk.transpose().reshape([1, -1, 6])

        # TODO: 将完成还原的逻辑参数序列反推回xy坐标序列
        self.fusion_xy = self.fusion_re.copy()
        return

    def sceneEnhace(self):
        '''对泛化场景在速度维度进行数据增强
        1. 离群点去除——在加速度维度上根据加速度上下界确定离群点位置，再利用均值滤波完成离群点去除
        2. 去噪——基于小波变换，对速度进行去噪
        3. 还原——基于完成数据增强的速度数据进行场景xy坐标还原，并基于还原的坐标差分得到一版新的速度、加速度数据（保持数据一致性）
        '''
        # TODO: 若略去数据强化阶段 ？为啥这样写
        if not self.need_repair:
            self.fusion_xy_repair = self.fusion_xy.copy()
            self.fusion_v_repair = self.fusion_v.copy()
            self.fusion_v_denosie = self.fusion_v.copy()
            self.fusion_a_repair = self.fusion_a.copy()
            self.fusion_j_repair = self.fusion_j.copy()
            print("跳过数据强化环节")
            np.save('./processed_by_zwt/output/cutin3.npy', self.fusion_xy)
            print("Saved the sample scenarios")

            # TODO: jerk值分析
            ego_x_jerk = jerkAnalysis(self.fusion_j[:, :, 0])
            ego_y_jerk = jerkAnalysis(self.fusion_j[:, :, 1])
            cutin_x_jerk = jerkAnalysis(self.fusion_j[:, :, 2])
            cutin_y_jerk = jerkAnalysis(self.fusion_j[:, :, 3])
            pre_x_jerk = jerkAnalysis(self.fusion_j[:, :, 4])
            pre_y_jerk = jerkAnalysis(self.fusion_j[:, :, 5])
            print('增强前的极端突变值比例：{},{},{},{},{},{}'.format(ego_x_jerk, ego_y_jerk, cutin_x_jerk, cutin_y_jerk, pre_x_jerk, pre_y_jerk))

            # TODO: 离群点分析
            self.fusion_v_outliar = self.fusion_v.copy()  # 深拷贝
            ego_x_outliar = outliarRemoval(self.fusion_a[:, :, 0], self.fusion_v_outliar[:, :, 0], self.lon_bound)
            ego_y_outliar = outliarRemoval(self.fusion_a[:, :, 1], self.fusion_v_outliar[:, :, 1], self.lat_bound)
            cutin_x_outliar = outliarRemoval(self.fusion_a[:, :, 2], self.fusion_v_outliar[:, :, 2], self.lon_bound)
            cutin_y_outliar = outliarRemoval(self.fusion_a[:, :, 3], self.fusion_v_outliar[:, :, 3], self.lat_bound)
            pre_x_outliar = outliarRemoval(self.fusion_a[:, :, 4], self.fusion_v_outliar[:, :, 4], self.lon_bound)
            pre_y_outliar = outliarRemoval(self.fusion_a[:, :, 5], self.fusion_v_outliar[:, :, 5], self.lat_bound)
            self.fusion_v_outliar = np.dstack((ego_x_outliar[0], ego_y_outliar[0], cutin_x_outliar[0], cutin_y_outliar[0], pre_x_outliar[0], pre_y_outliar[0]))
            # 维度[scenario_nums, 125, 6]
            print('增强前的动力学检查结果：{},{},{},{},{},{}'.format(ego_x_outliar[1:], ego_y_outliar[1:],
                  cutin_x_outliar[1:], cutin_y_outliar[1:], pre_x_outliar[1:], pre_y_outliar[1:]))
            return

        # TODO: 轨迹数据增强（旺哥PartC）
        noSolution_count = 0  # 记录优化模型无法求解的个数
        all_outlier_record = pd.DataFrame()  # 记录所有离群点
        for scene_ind in tqdm(range(self.fusion_xy.shape[0])):
            ego_x = self.fusion_xy[scene_ind, :, 0]
            ego_y = self.fusion_xy[scene_ind, :, 1]
            ego_vx = self.fusion_v[scene_ind, :, 0]
            ego_vy = self.fusion_v[scene_ind, :, 1]
            # 另外两辆背景车分别与主车组成pairs进行数据增强
            for i in range(2):
                cutin_x = self.fusion_xy[scene_ind, :, 2 * (i + 1)]
                cutin_y = self.fusion_xy[scene_ind, :, 2 * (i + 1) + 1]
                cutin_vx = self.fusion_v[scene_ind, :, 2 * (i + 1)]
                cutin_vy = self.fusion_v[scene_ind, :, 2 * (i + 1) + 1]
                # 将泛化场景数据的格式与PartC代码对齐
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
                # 进行上下行方向的统一
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
                out_trj = pair_cf_coord_cal(0, df_ego, 1, df_cutin, 0, all_outlier_record)  # 返回的为主车与另一辆车的dataframe
                if not out_trj.empty:
                    df_final = out_trj.loc[:, ['local_veh_id', 'length', 'local_time', 'filter_pos_x', 'filter_speed_x',
                                           'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']]
                    # 将主车与对手车相关信息从df提取至np.array中
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
                    # 相关数据组合存入对应self.fusion_repair
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
                            self.fusion_v_repair = temp_array_v.reshape([1, -1, 6])  # 对张量(4, 125)转置为(125, 4)，再reshape为(1, 125, 4)
                            self.fusion_a_repair = temp_array_a.reshape([1, -1, 6])
                else:
                    noSolution_count += 1
                    break
        print('无法通过数据增强实现场景修复的场景比例：{}%'.format(100 * (noSolution_count / self.fusion_xy.shape[0])))
        '''
        # TODO: 去噪
        ego_x_denoise = wavefilter(self.fusion_v_outliar[:, :, 0])
        ego_y_denoise = wavefilter(self.fusion_v_outliar[:, :, 1])
        cutin_x_denoise = wavefilter(self.fusion_v_outliar[:, :, 2])
        cutin_y_denoise = wavefilter(self.fusion_v_outliar[:, :, 3])
        self.fusion_v_denosie = np.dstack((ego_x_denoise, ego_y_denoise, cutin_x_denoise, cutin_y_denoise))

        # 速度/加速度曲线可视化
        isShow = 0
        if isShow:
            for i in range(self.fusion_v_denosie.shape[0]):
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
                plt.ion()
                plt.subplot(2, 1, 1)
                plt.title('主车速度（上）及加速度（下）曲线')
                plt.plot(self.fusion_v[i, :, 0])
                plt.ylim([0, 50])
                plt.subplot(2, 1, 2)
                plt.plot(self.fusion_v[i, :, 3])
                plt.ylim([-8, 5])
                plt.pause(2)
                plt.clf()
                plt.ioff()

        # TODO: 还原
        for i in range(self.fusion.shape[0]):
            x1, x2 = [self.fusion_xy[i, 0, 0]], [self.fusion_xy[i, 0, 2]]
            y1, y2 = [self.fusion_xy[i, 0, 1]], [self.fusion_xy[i, 0, 3]]
            for j in range(1, self.fusion.shape[1]):
                x1.append(x1[j - 1] + self.fusion_v_denosie[i, j - 1, 0] * self.rate)
                y1.append(y1[j - 1] + self.fusion_v_denosie[i, j - 1, 1] * self.rate)
                x2.append(x2[j - 1] + self.fusion_v_denosie[i, j - 1, 2] * self.rate)
                y2.append(y2[j - 1] + self.fusion_v_denosie[i, j - 1, 3] * self.rate)
            temp_array = np.empty((len(x1), 4), dtype=np.float64)
            temp_array[:, 0] = x1
            temp_array[:, 1] = y1
            temp_array[:, 2] = x2
            temp_array[:, 3] = y2
            velocity, accer, jerk = get_speed_and_accer(temp_array, self.rate)
            if self.fusion_xy_repair.size > 0:
                self.fusion_xy_repair = np.concatenate((self.fusion_xy_repair, temp_array.reshape([1, -1, 4])), axis=0)
                self.fusion_v_repair = np.concatenate((self.fusion_v_repair, velocity.transpose().reshape([1, -1, 4])), axis=0)
                self.fusion_a_repair = np.concatenate((self.fusion_a_repair, accer.transpose().reshape([1, -1, 4])), axis=0)
                self.fusion_j_repair = np.concatenate((self.fusion_j_repair, jerk.transpose().reshape([1, -1, 4])), axis=0)
            else:
                self.fusion_xy_repair = temp_array.reshape([1, -1, 4])
                self.fusion_v_repair = velocity.transpose().reshape([1, -1, 4])  # 对张量(4, 125)转置为(125, 4)，再reshape为(1, 125, 4)
                self.fusion_a_repair = accer.transpose().reshape([1, -1, 4])
                self.fusion_j_repair = jerk.transpose().reshape([1, -1, 4])
        '''
        np.save('./processed_by_zwt/output/cutin3_processed.npy', self.fusion_xy_repair)
        print("Saved the processed sample scenarios")
        self.fusion_v_denosie = self.fusion_v.copy()
        return

    def reconEval(self, orig_input):
        """重建能力评价，使用测试集/训练集完整通过Encoder→Decoder得到的重建场景与原始输入进行比较
        评价指标：平均位移误差ADE、最终位移误差FDE

        :param orig_input: self.fusion_xy对应的原始输入
        """
        evalADE, evalFDE = [], []
        for i in range(self.fusion.shape[0]):  # 每个场景单独计算ADE、FDE
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
        """
        TODO 和PPT上的判断标准不一致？
        功能性检查，确定场景是否为cutin场景
        判断准则：1. 横向运动渐趋平缓时，两车横向距离小于1m；2. 最大横向间距大于2m；3.横向运动渐趋平缓后，后车的最大THW小于5s
        """
        # 计算主车速度
        v_ego = self.fusion_v_repair[:, :, 0]
        if not self.isXy:
            delta_y = self.fusion_re[:, :, 1]
            delta_x = self.fusion_re[:, :, 0]
        else:
            delta_y = self.fusion_xy_repair[:, :, 1] - self.fusion_xy_repair[:, :, 3]
            delta_x = self.fusion_xy_repair[:, :, 0] - self.fusion_xy_repair[:, :, 2]
        # 有效性评价
        vaild_count = 0
        c1, c2, c3 = 0, 0, 0
        for i in range(delta_y.shape[0]):
            y = list(delta_y[i, :])
            v = list(v_ego[i, :])
            x = list(delta_x[i, :])
            flag1, flag2, flag3 = True, True, False
            # C1，最小横向距离小于1.5m（换道成功）
            if abs(y[-1]) > 1.5:
                flag1 = False
            else:
                c1 += 1
            # C2，最大横向距离大于1.5m（从相邻车道执行换道）
            if abs(y[0]) < 1.5:
                flag2 = False
            else:
                c2 += 1
            # C3，完成变道后（横向距离小于1.5m），THW小于5s(该换道行为对主车运行有影响)
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
        '''功能性检查，确定场景是否为cutin场景
        判断准则：1. 横向运动渐趋平缓时，两车横向距离小于1m；2. 最大横向间距大于2m；3.横向运动渐趋平缓后，后车的最大THW小于5s
        '''
        # 计算主车速度
        v_ego = self.fusion_v_denosie[:, :, 0]
        if not self.isXy:
            delta_y = self.fusion_re[:, :, 1]
            delta_x = self.fusion_re[:, :, 0]
        else:
            delta_y = self.fusion_xy_repair[:, :, 1] - self.fusion_xy_repair[:, :, 3]
            delta_x = self.fusion_xy_repair[:, :, 0] - self.fusion_xy_repair[:, :, 2]
        # 有效性评价
        vaild_count = 0
        c1, c2, c3 = 0, 0, 0
        for i in range(delta_y.shape[0]):
            y = list(delta_y[i, :])
            v = list(v_ego[i, :])
            x = list(delta_x[i, :])
            flag1, flag2, flag3 = True, True, True
            # C1，最小横向距离小于1.5m且持续时间超过1s（换道成功）
            # 默认最小横向距离出现在最后一帧，检查最后1s内主车和对手车横向距离是否都在1.5m内
            for j in range(len(y) - 1, len(y) - 26, -1):
                if abs(y[j]) > 1.5:
                    flag1 = False
                    break
            if flag1:
                c1 += 1
            # C2，最大横向距离大于2m（从相邻车道执行换道）
            # 默认最大横向距离出现在初始帧
            if abs(y[0]) < 1.5:
                flag2 = False
            else:
                c2 += 1
            # C3，完成变道后（横向距离小于1.5m），THW小于5s(该换道行为对主车运行有影响)
            # 依然使用主车与对手车横向距离小于1.5m作为完成cut in的过程的标志，判断此后两车之间的THW是否一直小于5s
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
            # 验证交集
            if flag1 and flag2 and flag3:
                vaild_count += 1
                self.realFlag[i] = 1
        result = [vaild_count / delta_y.shape[0], c1 / delta_y.shape[0], c2 / delta_y.shape[0], c3 / delta_y.shape[0]]
        print('C: {0[0]}, C1: {0[1]}, C2: {0[2]}, C3: {0[3]}'.format(result))
        return

    def isReal(self):
        '''cutin泛化场景的真实性检查，检查主要包括两部分：车辆动力学约束与jerk值分析
        车辆动力学约束：主要关注车辆的加减速是否符合加减速阈值——相较于sceneEnhance()，本方法仅涉及检查评价，不做进一步修复
        jerk值分析：1. 极端突变值——绝对值大于15；2.符号反转频率——一秒内符号反转次数超过1
        '''
        # TODO: 车辆动力学检查
        ego_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 0], self.fusion_v_repair[:, :, 0], self.lon_bound, flag=False)
        ego_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 1], self.fusion_v_repair[:, :, 1], self.lat_bound, flag=False)
        cutin_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 2], self.fusion_v_repair[:, :, 2], self.lon_bound, flag=False)
        cutin_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 3], self.fusion_v_repair[:, :, 3], self.lat_bound, flag=False)
        pre_x_outliar = outliarRemoval(self.fusion_a_repair[:, :, 4], self.fusion_v_repair[:, :, 4], self.lon_bound, flag=False)
        pre_y_outliar = outliarRemoval(self.fusion_a_repair[:, :, 5], self.fusion_v_repair[:, :, 5], self.lat_bound, flag=False)
        print('增强后的动力学检查结果：{},{},{},{},{},{}'.format(ego_x_outliar[1:], ego_y_outliar[1:],
              cutin_x_outliar[1:], cutin_y_outliar[1:], pre_x_outliar[1:], pre_y_outliar[1:]))

        # TODO: jerk值分析
        ego_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 0])
        ego_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 1])
        cutin_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 2])
        cutin_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 3])
        pre_x_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 4])
        pre_y_jerk = jerkAnalysis(self.fusion_j_repair[:, :, 5])
        print('增强后的极端突变值比例：{}, {}, {}, {}, {}, {}'.format(ego_x_jerk, ego_y_jerk, cutin_x_jerk, cutin_y_jerk, pre_x_jerk, pre_y_jerk))
        return

    def isVaild(self, test, test_xy):
        '''有效性检查，关注场景关键参数分布，
        关键运行参数分布：主车主要关注纵向速度变化，对手车主要关注横向速度变化
        对于时间序列相似性评价，对于场景数据，重点关注其变化趋势的相似性（对于幅度不关心，幅度由车辆动力学进行约束）
        变化趋势的相似性采用皮尔逊相似系数去量化（三种思路：直接输入速度序列/模式距离/振幅距离）
        采用模式距离作为皮尔逊相似系数的输入，因为作为泛化，并不要求两个场景关键参数完全一致，而是需要保证基本的相同变化趋势
        【只能运用于测试集检查，即存在真值才能进行相关性计算】

        :param test: 测试集场景数据（逻辑场景参数序列）[num_scene, len_scene, 4] -> [delta_x, delta_y, vx_cutin, vy_cutin]
        :param test_xy: 测试集场景数据（坐标序列）[num_scene, len_scene, 4] -> [ego_x, pre_y, cutin_x, cutin_y]
        '''
        def getPattern(nums):
            '''将序列转换为模式序列'''
            nums = nums[:, ::3]
            diff_num = np.diff(nums, axis=1)
            pattern = np.where(diff_num > 0, 1, diff_num)
            pattern = np.where(pattern < 0, -1, pattern)
            return pattern
        '''
        # 主车速度(泛化值及真值)
        v_ego = np.array([])
        for i in self.fusion_xy[:, :, 0]:
            temp_v = np.diff(i, axis=0)  # 计算每帧下的速度
            temp_v = (temp_v / self.rate)
            # temp_v = wavefilter(temp_v)
            if v_ego.size > 0:
                v_ego = np.concatenate((v_ego, temp_v.reshape(1, -1)), axis=0)
            else:
                v_ego = temp_v.reshape(1, -1).copy()
        pattern_ego = getPattern(v_ego)
        true_v_ego = np.array([])
        for i in test_xy[:, :, 0]:
            temp_v = np.diff(i, axis=0)  # 计算每帧下的速度
            temp_v = (temp_v / self.rate)
            # temp_v = wavefilter(temp_v)
            if true_v_ego.size > 0:
                true_v_ego = np.concatenate((true_v_ego, temp_v.reshape(1, -1)), axis=0)
            else:
                true_v_ego = temp_v.reshape(1, -1).copy()
        pattern_true_ego = getPattern(true_v_ego)
        '''
        # 可视化验证
        # denoise = self.wavefilter(self.fusion_re[:, :, 0])
        for i in range(self.fusion_v_repair.shape[0]):
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
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
        # 主车速度变化的相关性
        pattern_dis_ego = np.mean(np.abs(pattern_ego - pattern_true_ego), axis=1)  # 主车速度的模式距离
        ego = []
        for i in range(v_ego.shape[0]):
            r, p = stats.pearsonr(pattern_true_ego[i, :], pattern_ego[i, :])
            ego.append([r, p])
        # 对手车横纵向速度相关性
        pattern_dis_cutin = []  # 对手车横纵向速度的模式距离
        pearson_cutin = []
        for i in range(2, 4):
            pattern_true = getPattern(test[:, :, i])
            pattern_fusion = getPattern(self.fusion[:, :, i])
            pattern_dis_cutin.append(np.mean(np.abs(pattern_fusion - pattern_true), axis=1))
            pearson_cutin.append([stats.pearsonr(pattern_true[i, :], pattern_fusion[i, :]) for i in range(test.shape[0])])
        print(np.mean(pattern_dis_ego), np.mean(pattern_dis_cutin[0]), np.mean(pattern_dis_cutin[1]))
        return
    
    def getMttc(self):
        '''获得场景集合内每个场景的最小MTTC
        '''
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
    """ 
    # 泛化评价（泛化数据集）
    trainScene = Reparir_cutin(fusion_path='./argoverse-data/cutin1_v_singleD_norm.npy', lims=lims)
    fusionScene = Reparir_cutin(fusion_path='./samples/x_f_s.0.npy', lims=lims)
    print(trainScene.isCutin())
    print(fusionScene.isCutin())
    PI = fusionScene.isReal(thresholdList=threshold)
    # print(PI)

    """
    '''
    # 有效性评价（原始数据）
    if isOrigin:
        testScene = Repair_cutin(fusion_path='./VAE_for_GAN/argoverse-data/cutin_xy_singleD_proceseed_norm_train.npy', isXy=isXy, lims=lims)
    else:
        testScene = Repair_cutin(fusion_path='./argoverse-data/cutin_xy_singleD_proceseed_norm_test.npy', isXy=isXy, lims=lims)
    print(testScene.isCutin())
    testScene.isReal()
    '''
    '''
    # 针对泛化效果的评价
    isSample = True
    if not isSample:
        # 针对重建效果的评价
        inputScene = Repair_cutin(fusion_path='./VAE_for_GAN/argoverse-data/cutin3_xy_singleD_proceseed_norm_train.npy', lims=lims)
        reconScene = Repair_cutin(fusion_path='./VAE_for_GAN/test/data_for_vaild.npy', lims=lims) 
        eval = reconScene.reconEval(inputScene.fusion_xy)
        print('ADE: {}, FDE: {}'.format(str(eval[0]), str(eval[1])))
        print(reconScene.isCutin())
        reconScene.isReal()
    else:
        originalScene = Repair_cutin(fusion_path='./VAE_WGAN/samples/data_without_latent_cons.npy', lims=lims)
        originalScene.isReal()
        sampleScene = Repair_cutin(fusion_path='./VAE_WGAN/samples/data_with_latent_cons.npy', lims=lims)
        sampleScene.isReal()

    '''
    '''
    # 针对泛化场景多样性的评价
    inputScene = Repair_cutin(fusion_path='./VAE_for_GAN/argoverse-data/cutin3_xy_singleD_proceseed_norm_train.npy', lims=lims)
    sampleScene = Repair_cutin(fusion_path='./VAE_WGAN/samples/data_with_latent_cons.npy', lims=lims)
    # 绘制MTTC分布
    ax1 = plt.subplot(111)
    input_mttc = inputScene.fusion_mttc
    sample_mttc = sampleScene.fusion_mttc
    sns.kdeplot(data=input_mttc, label="Input Scenarios", fill=True, ax=ax1)
    sns.kdeplot(data=sample_mttc, label="Sample Scenarios", fill=True, ax=ax1)
    ax1.legend()
    ax1.set_ylabel('MTTC')
    plt.savefig('./VAE_WGAN/image/minMTTC.png', dpi=600)
    # 绘制THW分布
    plt.clf()
    ax2 = plt.subplot(111)
    input_thw = inputScene.fusion_thw
    sample_thw = sampleScene.fusion_thw
    sns.kdeplot(data=input_thw, label="Input Scenarios", fill=True, ax=ax2)
    sns.kdeplot(data=sample_thw, label="Sample Scenarios", fill=True, ax=ax2)
    ax2.legend()
    ax2.set_ylabel('minTHW')
    plt.savefig('./VAE_WGAN/image/minTHW.png', dpi=600)
    plt.tight_layout()
    '''
    # 针对车辆速度加速度平滑程度的评价
    # sampleScene = Repair_cutin(fusion_path='./VAE_WGAN/samples/data_with_latent_cons.npy', lims=lims)
    sampleScene_orig = Repair_cutin(fusion_path='./VAE_WGAN/output/cutin3.npy', isNorm=False, need_repair=False)
    sampleScene_processed = Repair_cutin(fusion_path='./VAE_WGAN/output/cutin3_processed.npy', isNorm=False, need_repair=False)

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
