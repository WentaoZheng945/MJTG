import os.path
import numpy as np
from subFunctions import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DangerFilter():
    def __init__(self, lims=None, frame_rate=0.04, isNorm=True, fusion_path='', save_path='./processed/input_data', save=True):
        self.rate = frame_rate
        self.isNorm = isNorm
        if lims is None:
            lims = []
        self.lims = lims
        self.save_path = save_path
        self.save = save
        self.fusion = np.load(fusion_path, allow_pickle=True)
        self.fusion_re = np.array([])
        self.fusion_xy = np.array([])
        self.fusion_v = np.array([])
        self.fusion_a = np.array([])
        self.fusion_j = np.array([])
        self.dangerFlag = [0] * self.fusion.shape[0]
        self.fusion_mttc = []
        self.sceneRecoverPosition()
        self.getMTTC()
        if save:
            self.split_train_dataset()

    def sceneRecoverPosition(self):
        print('{0}个场景正在还原'.format(self.fusion.shape[0]))
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
            # [ego_x, ego_y, cutin_x, cutin_y, pre_x, pre_y], shape(6, 125)
            if self.fusion_v.size > 0:
                self.fusion_v = np.concatenate((self.fusion_v, velocity.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_a = np.concatenate((self.fusion_a, accer.transpose().reshape([1, -1, 6])), axis=0)
                self.fusion_j = np.concatenate((self.fusion_j, jerk.transpose().reshape([1, -1, 6])), axis=0)
            else:
                self.fusion_v = velocity.transpose().reshape([1, -1, 6])
                self.fusion_a = accer.transpose().reshape([1, -1, 6])
                self.fusion_j = jerk.transpose().reshape([1, -1, 6])

        self.fusion_xy = self.fusion_re.copy()
        print('[*] 还原结束')
        return


    def getMTTC(self):
        pbar_scenarios = tqdm(total=self.fusion_xy.shape[0], desc='[scenarios]')
        for i in range(self.fusion_xy.shape[0]):
            cur_mttc = []
            for j in range(self.fusion_xy.shape[1] - 1, -1, -1):
                if abs(self.fusion_xy[i, j, 1] - self.fusion_xy[i, j, 3]) <= 1.5:
                    dist = abs(self.fusion_xy[i, j, 0] - self.fusion_xy[i, j, 2])
                    # delta_v = abs(self.fusion_v_repair[i, j, 0]) - abs(self.fusion_v_repair[i, j, 2])
                    delta_v = self.fusion_v[i, j, 0] - self.fusion_v[i, j, 2]
                    delta_v2 = (self.fusion_v[i, j, 0] - self.fusion_v[i, j, 2]) ** 2
                    delta_a = self.fusion_a[i, j, 0] - self.fusion_a[i, j, 2]
                    if delta_a != 0:
                        t1 = (delta_v * (-1) - (delta_v2 + 2 * delta_a * dist) ** 0.5) / delta_a
                        t2 = (delta_v * (-1) + (delta_v2 + 2 * delta_a * dist) ** 0.5) / delta_a
                        if t1 > 0 and t2 > 0:
                            cur_mttc.append(min(t1, t2))
                        elif t1 * t2 <= 0 and max(t1, t2) > 0:
                            cur_mttc.append(max(t1, t2))
                        else:
                            cur_mttc.append(dist / abs(self.fusion_v[i, j, 0]))
                else:
                    # cur_mttc.append(float('inf'))
                    # cur_thw.append(float('inf'))
                    break
            try:
                self.fusion_mttc.append(min(cur_mttc))
            except ValueError:
                self.fusion_mttc.append(4)
            pbar_scenarios.update(1)
        self.fill_dangerFlag()
        return

    def fill_dangerFlag(self):
        for idx, item in enumerate(self.fusion_mttc):
            if item <= 1:
                self.dangerFlag[idx] = 1
        return

    def split_train_dataset(self):
        self.dangerFlag = np.array(self.dangerFlag)
        danger_index = self.dangerFlag == 1
        safe_index = self.dangerFlag == 0
        danger_npy = self.fusion[danger_index]
        safe_npy = self.fusion[safe_index]
        print(danger_npy.shape[0])
        print(safe_npy.shape[0])
        np.save(os.path.join(self.save_path, 'cutin3_xy_singleD_proceseed_norm_train_danger_1.npy'), danger_npy)
        np.save(os.path.join(self.save_path, 'cutin3_xy_singleD_proceseed_norm_train_safe_1.npy'), safe_npy)
        print('[*] 划分数据集并成功保存')
        return


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
    trainScene = DangerFilter(lims=lims, frame_rate=0.04, isNorm=True, fusion_path=global_trainScene_path)
