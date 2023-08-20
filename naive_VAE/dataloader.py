import numpy as np
from torch.utils.data import Dataset
import random
import os


class Sequence_Dataset(Dataset):
    """
    针对待训练的数据集，定义Sequence_Dataset类对数据进行储存（继承自torch.utils.data.Dataset()类，且必须复写__getitem__()方法）
    其中__getitem__()与__len__()方法是对该Sequence_Dataset类的魔法方法实现
    如__len__()方法，只有在Sequence_Dataset类中实现了该方法，Sequence_Dataset类对象才能使用len()方法获得类的长度
    【自定义的类与python中的基础类型（int,list,str等）具有同等地位，基础类能使用len()也是因为它们的类实现了__len__()方法】
    如果是带标签的数据，把data和label分别初始化，__getitem__也分别返回data[index]和label[index]
    """
    def __init__(self, data_path):
        """
        魔法方法__init__()，当Sequnece_Dataset类被建立，立即执行该方法，完成各类数据的加载（作为该类的属性）
        """
        self.data_normal = np.load(data_path)  # 输出尺寸(num, 125, 4)

        self.indices = range(len(self))

    def __getitem__(self, index):
        """
        __getitem__()方法是Dataset类的核心，其作用是接收一个索引，返回一个样本。重写的目的在于根据需求确定如何根据索引读取数据
        """
        # ref_index = random.randint(0, len(self.data_normal) - 1)  # 生成事故数据索引范围内的随机整数，类似shuffle作用
        return self.data_normal[index]

    def __len__(self):
        return len(self.data_normal)
