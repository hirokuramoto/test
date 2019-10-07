# 世代交代モデルの定義

from abc import ABCMeta, abstractmethod
import random
import math
import numpy as np

class Crossover(metaclass = ABCMeta):
    """交叉方法のベース
    """
    def __init__(self, generate_size):
        """constractor
        Args :
            generate_size (int) : 交叉によって生成する個体数
        """
        self._generate_size = generate_size

    @abstractmethod
    def crossover(self, individual_set, parents_set):
        """交叉を実行する
        Args:
            individual_set (np.array) : 個体の2次元配列
            parents_set (np.array)  : 個体集団の中で使用する親個体
        Returns :
            children_set (np.array)   : 生成した子個体
        """
        pass

class Simplex(Crossover):
    def crossover(self, individual_set, parents_set):
        """次元数+1個体から子個体を生成する
        """
        # 親個体を抽出して2次元配列化
        matrix = np.array([individual_set[i] for i in parents_set])
        # 列ごと(axis=0)の平均値を求める
        center = matrix.mean(axis = 0)
        # 設計変数の数を抽出
        dimension = len(center)

        alpha = math.sqrt(dimension + 2)
        matrix = center + alpha * (matrix - center)

        # 空配列を用意
        children_set = np.empty((0, dimension), dtype = np.float64)

        for _ in range(self._generate_size):
            # 子1個体の遺伝子を格納する空配列を用意
            gene = np.zeros(dimension)

            for k, (vector1, vector2) in enumerate(zip(matrix, matrix[1:])):
                r_k = random.uniform(0., 1.) ** (1./(k+1))
                child = r_k * (vector1 - vector2 + gene)
            gene += matrix[-1]
            child = np.append(children_set, np.array([[child]]))
        return children_set


if __name__ == "__main__":
    from generator import *
    from evaluator import *
    from individual_selector import *

    generator = Generator(10, 0, 3, 5)
    individual_set = generator.generate()

    evaluator = Evaluator(individual_set)
    evaluate_set = evaluator.evaluate()

    elite_set = EliteSelector(2).select(individual_set, evaluate_set)
    roulette_set = RouletteSelector(4).select(individual_set, evaluate_set)

    test = Simplex(3).crossover(individual_set, roulette_set)



    print(individual_set)
    print(evaluate_set)
    print(elite_set)
    print(roulette_set)
    print(test)
