# ガウスパラメータ，正則化パラメータの決定のためにCV値を求める

import numpy as np
from .call_fortran import *

class LeaveOneOut(object):
    def __init__(self, beta, penalty, design_data, object_data):
        """ガウスカーネルを使った予測値ベクトルを返す
        Args :
            beta (float) : ガウスパラメータ　β
            penalty (float) : 正則化パラメータ　λ
            design_data (np.array) : 標準化済みの設計変数配列
            object_data (np.array) : 訓練データの結果配列
        Returns :

        """

        self._beta = beta
        self._penalty = penalty
        self._design_data = design_data
        self._object_data = object_data

    def cross_validation(self, number):
        """constractor
        Args :
            number (int) : 何番目の目的関数について調べるか
        """
        # テストデータの行数（個体数）を取得
        data_size = self._design_data.shape[0]

        # 設計変数配列を取得
        design_data = self._design_data

        # 訓練データの結果配列を取得
        object_data = self._object_data

        beta = self._beta
        n_param = design_data.shape[1]


        # グラム行列の計算
        #gram_matrix = np.identity(data_size)
        #for i in range(data_size):
        #    for k in range(i + 1, data_size):
        #        gram_matrix[i][k] = np.exp(-1 * self._beta * np.inner(design_data[i,] - design_data[k,], design_data[i,] - design_data[k,]))
        #        gram_matrix[k][i] = gram_matrix[i][k]

        # Fortranのグラム行列計算用サブルーチンを呼び出す
        # 渡す行列データを転置（Fortranは列majorのため）
        design_data = design_data.T
        gram_matrix = np.identity(data_size)
        call = CallFortran(data_size, n_param, beta, design_data, gram_matrix)
        call.call_fortran()

        # 重みベクトルの計算
        i_mat = np.identity(data_size)
        alpha_vector = np.dot(np.linalg.inv(gram_matrix + self._penalty * i_mat), object_data)

        # 予測値の計算
        predict_vector = np.dot(gram_matrix, alpha_vector)

        # H行列の計算
        i_mat = np.identity(data_size)
        h_matrix = np.dot(np.linalg.inv(gram_matrix + self._penalty * i_mat), gram_matrix)

        # CV値の計算
        cv_value = 0.
        for i in range(data_size):
            cv_value += ((object_data[i, number] - predict_vector[i, number])/(1 - h_matrix[i][i])) ** 2

        return cv_value


if __name__ == '__main__':
    from standard_data import StandardData

    # テストデータを取得
    test = StandardData(5,2)
    data = test.standard("result.csv")

    design = np.array(data[0:, 0:5])
    object = np.array(data[0:, 5:7])

    test = LeaveOneOut(0.1, 0.001, design, object)
    test1 = test.cross_validation(0)
    print(test1)
