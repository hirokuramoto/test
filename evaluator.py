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
        """Rosenbrock関数で個体集団を評価
        Returns :
            np.array（1次元配列）：評価値配列
        """

        # individual_setの行数（個体数）を取得
        size = self._individual_set.shape[0]

        evaluate_set = np.array([], dtype = np.float64)
        for i in range(size):
            result = 100.0 * (self._individual_set[i,1] - self._individual_set[i,0]**2) ** 2 + (self._individual_set[i,0] - 1)** 2
            evaluate_set = np.append(evaluate_set, result)
        return evaluate_set







if __name__ == "__main__":
    generator = Generator(10, 0, 2, 3)
    # 個体集団の2次元配列の取得
    individual_set = generator.generate()

    evaluator = Evaluator(individual_set)
    # 個体集団の評価値配列（1次元）の取得
    test = evaluator.evaluate()

    print(individual_set)
    print(test)
