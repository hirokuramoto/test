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

class BLX_alpha(Crossover):
    def __init__(self, generate_size, alpha = 0.3):
        super(BLX_alpha, self).__init__(generate_size) # generate_sizeはスーパークラスの__init__()で定義したものを再利用
        self._alpha = alpha

    def crossover(self, individual_set, parents_set):
        """2個体から子個体を生成する
        """

        matrix = np.array([individual_set[i] for i in parents_set[:2]])
        return matrix



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

    test = BLX_alpha(2).crossover(individual_set, roulette_set)



    print(individual_set)
    print(evaluate_set)
    print(elite_set)
    print(roulette_set)
    print(test)
