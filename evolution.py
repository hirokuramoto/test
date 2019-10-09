# 世代交代しながら進化

from crossover import *
from evaluator import *
from individual_selector import *
from generation_selector import *
import numpy as np

class Evolution(object):
    def __init__(self, evaluator, crossover, parents_selector, generation_selector):
        """constractor
        Args :
            evaluator : 評価関数
            crossover : 交叉
            parents_selector : 親の選択セレクタ（エリート/ルーレット）
            generation_selector : 次世代の選択セレクタ（JGG）
        """
        self._evaluator = evaluator
        self._crossover = crossover
        self._parents_selector = parents_selector
        self._generation_selector = generation_selector
        self._individual_set = None
        self._evaluate_set = None

    def generate_individuals(self, generator):
        """初期個体の生成
        Args :
            generator : 生成関数
        """
        self._individual_set = generator.generate()

    def evaluate_individuals(self, evaluator):
        """初期個体の評価
        Args :
            evaluator : 評価関数
        """
        evaluator = Evaluator(self._individual_set)
        self._evaluate_set = evaluator.evaluate()


    def change_generation(self):
        """世代交代を行う
        """
        # 親個体のindex配列取得
        parents_index = self._parents_selector.select(self._individual_set, self._evaluate_set)

        # 交叉
        children_set = self._crossover.crossover(self._individual_set, parents_index)

        # 親＋子個体全てから次世代を選ぶ
        self._individual_set = self._generation_selector.select(self._individual_set, parents_index, children_set)


    @property
    def individual_set(self):
        return self._individual_set

    @property
    def evaluate_set(self):
        return self._evaluate_set


if __name__ == "__main__":
    from generator import *
    from evaluator import *
    from individual_selector import *
    from crossover import *

    dimension = 3

    # 初期個体の生成
    generator = Generator(10, 0, 3, 5)
    evaluator = Evaluator()

    # 初期個体の生成
    evolution = Evolution(evaluator, Simplex(dimension * 10), RouletteSelector(dimension + 1), JGG())
    evolution.generate_individuals(generator)
    evolution.evaluate_individuals(evaluator)
    print(evolution.individual_set)
    print(evolution.evaluate_set)
