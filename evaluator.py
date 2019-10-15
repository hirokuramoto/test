# 評価関数による評価

import numpy as np
from abc import ABCMeta, abstractmethod
from generator import Generator
from standard_data import StandardData
from leave_one_out import LeaveOneOut


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
    def __init__(self, data, design_variables, objective_variables, num):
        # CrossValidation を使うときは引数が１つ増えるので注意
        self._data = data       # テストデータ
        self._design_variables = design_variables
        self._objective_variables = objective_variables
        self._num = num

    def _evaluate_function(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
            data : 訓練データの2次元配列
        Returns :
            np.array（1次元配列）：評価値配列
        """
        # 設計変数の個数指定
        design_variables = self._design_variables

        # 目的関数の個数指定
        objective_variables = self._objective_variables

        # 何番目の目的関数について調べるか
        num = self._num


        #テストデータのN数を取得
        data_size = self._data.shape[0]

        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]

        # テストデータの設計変数と目的関数を取得
        design = np.array(self._data[0:, 0:design_variables])
        object = np.array(self._data[0:, design_variables:-1])

        evaluate_set = np.array([], dtype = np.float64)
        for i in range(size):
            x = LeaveOneOut(individual_set[i, 0], individual_set[i, 1], design, object)
            result = x.cross_validation(num)
            evaluate_set = np.append(evaluate_set, result)

        return evaluate_set







if __name__ == "__main__":
    generator = Generator(10, 0, 2, 100)
    # 個体集団の2次元配列の取得
    individual_set = generator.generate()

    # テストデータを取得
    data = StandardData(5, 2).standard("result.csv")

    evaluator = CrossValidation(data, 5, 2, 0)
    # 個体集団の評価値配列（1次元）の取得
    test = evaluator.evaluate(individual_set)

    print(individual_set)
    print(test)
