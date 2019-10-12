# ガウスカーネルによるカーネルリッジ回帰を行う

import numpy as np

class KernelRidge(object):
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

    def kernel_ridge(self):

        # テストデータの行数（個体数）を取得
        data_size = self._design_data.shape[0]

        # 設計変数配列を取得
        design_data = self._design_data

        # 訓練データの結果配列を取得
        object_data = self._object_data

        # グラム行列の計算
        gram_matrix = np.identity(data_size)
        for i in range(data_size):
            for k in range(i+1, data_size):
                gram_matrix[i][k] = np.exp(-1 * self._beta * np.inner(design_data[i,] - design_data[k,], design_data[i,] - design_data[k,]))
                gram_matrix[k][i] = gram_matrix[i][k]

        # 重みベクトルの計算
        i_mat = np.identity(data_size)
        alpha_vector = np.dot(np.linalg.inv(gram_matrix + self._penalty * i_mat), object_data)

        # 予測値の計算
        predict_vector = np.dot(gram_matrix, alpha_vector)

        return gram_matrix#predict_vector

if __name__ == '__main__':
    from standard_data import StandardData

    # テストデータを取得
    test = StandardData(5,2)
    data = test.standard("result.csv")

    design = np.array(data[0:, 0:5])
    object = np.array(data[0:, 5:7])

    test = KernelRidge(0.1, 0.001, design, object)
    test1 = test.kernel_ridge()
    print(test1)
