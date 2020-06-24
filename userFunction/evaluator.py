# 評価関数による評価
import subprocess
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from generator import Generator
#from .standard_data import StandardData
#from .leave_one_out import LeaveOneOut


class Evaluator(metaclass=ABCMeta):
    def evaluate(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
        Returns :
            np.array（1次元配列）：評価値配列
        """
        return self._evaluate_function(individual_set)

    @abstractmethod
    def _evaluate_function(self, individual_set):
        pass

class Benchmark(Evaluator):
    def _evaluate_function(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
        Returns :
            np.array（1次元配列）：評価値配列
        """
        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]
        # 設計変数ファイルを作成
        np.savetxt('./userFunction/pop_vars_eval.txt', individual_set, delimiter='\t', fmt='%.9f')
        # 計算実行
        cmd0 = './userFunction/mazda_mop ./userFunction/'
        subprocess.run(cmd0.split())

        # 目的関数ファイルの1列目を抽出（3車種の合計重量）
        data1 = np.loadtxt('./userFunction/pop_objs_eval.txt')
        weight = data1[:, 0]

        # 重量を記録
        if len(weight) > 300:
            pass
        else:
            df = pd.DataFrame(weight)
            df.to_csv('weight.csv', mode='a', header=False)

        # ペナルティを計算
        P  = 0.0     # ペナルティ切片
        Pc = 1000.0 # ペナルティ係数
        Pe = 2    # ペナルティ指数

        penalty = np.empty(size)
        data2 = np.loadtxt('./userFunction/pop_cons_eval.txt')
        for i in range(size):
            sigma = 0
            for j in range(54):
                if data2[i, j] >= 0:
                    data2[i, j] = 0
                else:
                    pass
                sigma = sigma + (data2[i, j] ** Pe)
            penalty[i] = P + Pc * sigma

        evaluate_set = weight + penalty
        return evaluate_set


class Rosenbrock(Evaluator):
    def _evaluate_function(self, individual_set):

        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]

        evaluate_set = np.array([], dtype = np.float64)
        for i in range(size):
            result = 100.0 * (individual_set[i,1] - individual_set[i,0]**2) ** 2 + (individual_set[i,0] - 1)** 2
            evaluate_set = np.append(evaluate_set, result)
        return evaluate_set


class CrossValidation(Evaluator):
    def _evaluate_function(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
            data : 訓練データの2次元配列
        Returns :
            np.array（1次元配列）：評価値配列
        """

        # 訓練用データ
        design = 5              # 設計変数の数
        object = 2              # 目的関数の数
        num = 0                 # 何番目の目的関数について調べるか
        filename = "result.csv" # 訓練データのcsvファイル

        # 訓練用データの取り込み
        data = StandardData(design, object).standard(filename)

        #テストデータのN数を取得
        data_size = data.shape[0]

        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]

        # テストデータの設計変数と目的関数を取得
        design_data = np.array(data[0:, 0:design])
        object_data = np.array(data[0:, design-1:-1])

        evaluate_set = np.array([], dtype = np.float64)
        for i in range(size):
            x = LeaveOneOut(individual_set[i, 0], individual_set[i, 1], design_data, object_data)
            result = x.cross_validation(num)
            evaluate_set = np.append(evaluate_set, result)

        return evaluate_set

if __name__ == "__main__":

    boundary = 'bound.csv'

    test = Generator(boundary, 222, 300)

    # 個体集団の2次元配列の取得
    individual_set = test.generate()

    function = Benchmark()

    function.evaluate(individual_set)
