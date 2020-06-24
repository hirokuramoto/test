# 個体の生成

import random
import numpy as np
import pandas as pd

class Generator(object):
    def __init__(self, boundary, dimension, size):
        """constractor
        Args :
            boundary : パラメータ値の境界条件csvデータ（initial, lower, upper, range）
            dimension : パラメータ数
            size (int) : 個体数
        """
        self._boundary = boundary
        self._dimension = dimension
        self._size = size

    def generate(self):
        """個体集団を生成する
        Returns :
            np.array（2次元配列）：1行が1個体
        """

        # 訓練データの1次元配列化
        data = pd.read_csv(self._boundary, header = 0) # header = 0でヘッダー行を読み飛ばし
        lower_value = np.array(data.iloc[:, 1])
        upper_value = np.array(data.iloc[:, 2])
        range_value = np.array(data.iloc[:, 3])

        dimension = range_value.shape[0]

        #individual = np.array([], dtype = np.float64)
        individual_set = np.array([], dtype = np.float64)

        if dimension == self._dimension:
            return  np.array([[range_value[i] * np.random.rand() + lower_value[i] for i in range(dimension)] for _ in range(self._size)])
        else:
            print("dimensionの値が合いません")

if __name__ == "__main__":

    boundary = './userFunction/bound_test.csv'
    gene = Generator(boundary, 2, 100)
    individual_set = gene.generate()
    np.savetxt('individual_set.csv', individual_set, delimiter=',')
