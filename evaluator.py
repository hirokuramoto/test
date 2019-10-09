# 評価関数による評価

import numpy as np
from abc import ABCMeta, abstractmethod
from generator import Generator

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



if __name__ == "__main__":
    generator = Generator(10, 0, 2, 3)
    # 個体集団の2次元配列の取得
    individual_set = generator.generate()

    evaluator = Rosenbrock()
    # 個体集団の評価値配列（1次元）の取得
    test = evaluator.evaluate(individual_set)

    print(individual_set)
    print(test)
