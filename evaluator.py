# 評価関数による評価

import numpy as np
from generator import Generator

class Evaluator(object):
    def __init__(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
        """
        self._individual_set = individual_set

    def evaluate(self):
        """個体集団を評価
        Returns :
            np.array（1次元配列）：評価値配列
        """
        # 要素の合計（評価値）を返す
        evaluate_set = np.sum(self._individual_set, axis = 1)
        return evaluate_set







if __name__ == "__main__":
    generator = Generator(10, 0, 4, 3)
    # 個体集団の2次元配列の取得
    individual_set = generator.generate()

    evaluator = Evaluator(individual_set)
    # 個体集団の評価値配列（1次元）の取得
    test = evaluator.evaluate()

    print(individual_set)
    print(test)
