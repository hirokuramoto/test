# 世代交代モデルの定義

from abc import ABCMeta, abstractmethod
import random
import numpy as np
from individual_selector import *
from userFunction import evaluator

class GenerationSelector(metaclass=ABCMeta):
    """次世代に残す個体の選択方法のベース
    """

    @abstractmethod
    def select(self, individual_set, parents_index, children_set):
        """次世代に残す個体のリストを返す
        Args :
            individual_set : 個体の2次元配列
            parents_index : 親個体のindex配列
            children_set : 生成した子個体
        """
        pass
        # 親個体の集合（2次元配列）
        parents_set = np.array([individual_set[x] for x in parents_index])

class JGG(GenerationSelector):
    """Just Generation Gap による世代交代
    """

    def select(self, individual_set, parents_index, children_set, children_value):
        """子個体からエリート個体を選択し，親個体と入れ替える
        """
        # 子個体からエリート個体のindexを取得
        elite_index = EliteSelector(parents_index.size).select(children_set, children_value)

        # 個体入れ替え
        for p, e in zip(parents_index, elite_index):
            individual_set[p] = children_set[e]
        return individual_set




if __name__ == "__main__":
    from generator import *
    from evaluator import *
    from individual_selector import *
    from crossover import *

    generator = Generator(10, 0, 3, 5)
    individual_set = generator.generate()

    evaluator = Evaluator(individual_set)
    evaluate_set = evaluator.evaluate()

    elite_index = EliteSelector(2).select(individual_set, evaluate_set)
    roulette_index = RouletteSelector(4).select(individual_set, evaluate_set)
    children_set = Simplex(5).crossover(individual_set, roulette_index)

    print(individual_set)
    print(evaluate_set)
    print(elite_index)
    print(roulette_index)
    print(children_set)

    individual_set = JGG().select(individual_set, roulette_index, children_set)
    print(individual_set)
