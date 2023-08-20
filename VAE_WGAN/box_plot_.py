# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/19 10:37
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def boxplot_compare(y1, y2, x1, x2, columns):
    box1 = plt.boxplot(y1, positions=x1, patch_artist=True, showmeans=True, showfliers=False,
                       boxprops={"facecolor": "C0",
                                 "edgecolor": "grey",
                                 "linewidth": 0.5},
                       medianprops={"color": "k", "linewidth": 0.5},
                       meanprops={'marker': '+',
                                  'markerfacecolor': 'k',
                                  'markeredgecolor': 'k',
                                  'markersize': 5},
                       flierprops={'marker': 'o',
                                  'markerfacecolor': 'k',
                                  'markeredgecolor': 'k',
                                  'markersize': 5},
                       autorange=True)
    box2 = plt.boxplot(y2, positions=x2, patch_artist=True, showmeans=True, showfliers=False,
                       boxprops={"facecolor": "C1",
                                 "edgecolor": "grey",
                                 "linewidth": 0.5},
                       medianprops={"color": "k", "linewidth": 0.5},
                       meanprops={'marker': '+',
                                  'markerfacecolor': 'k',
                                  'markeredgecolor': 'k',
                                  'markersize': 5},
                       flierprops={'marker': 'o',
                                   'markerfacecolor': 'k',
                                   'markeredgecolor': 'k',
                                   'markersize': 5},
                       autorange=True)

    plt.xticks([1.5, 4.5, 7.5, 10.5], columns, fontsize=16)
    # plt.ylim(19.2, 20.1)
    # plt.xlim(6.5, 8.5)
    plt.ylabel('Scores', fontsize=16)
    plt.grid(axis='y', ls='--', alpha=0.8)

    # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
    plt.legend(handles=[box1['boxes'][0], box2['boxes'][0]], labels=['scores on real-world scenarios', 'scores on derived scenarios'], fontsize=16)

    plt.tight_layout()
    plt.savefig('./processed_by_zwt/image/boxplot_comfort.svg', dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    global_train_score_path = r'./processed_by_zwt/scores/train.csv'
    global_sample_score_path = r'./processed_by_zwt/scores/samples.csv'
    global_train_score = pd.read_csv(global_train_score_path)
    global_sample_score = pd.read_csv(global_sample_score_path)
    global_train_score = global_train_score[['Safety', 'Efficiency', 'Comfort', 'Total']]
    global_sample_score = global_sample_score[['Safety', 'Efficiency', 'Comfort', 'Total']]
    global_columns = global_sample_score.columns
    global_train_score = global_train_score.values
    global_sample_score = global_sample_score.values
    global_x1 = np.arange(1, 12, 3)
    global_x2 = global_x1 + 1
    boxplot_compare(global_train_score, global_sample_score, global_x1, global_x2, global_columns)
