# メインファイル

from generator import *
from evaluator import *
from crossover import *
from individual_selector import *
from generation_selector import *
import numpy as np

def test():
    # 遺伝子の値の最大値
    maximum = 10

    # 遺伝子の値の最小値
    minimum = 0

    # パラメータ数
    dimension = 2

    #　個体数
    size = 100

    # 繰り返し数
    generation_loop = 1000

    # 初期個体の生成
    generator = Generator(maximum, minimum, dimension, size)
    individual_set = generator.generate()
    print(individual_set)

    # 初期個体の評価
    evaluator = Evaluator(individual_set)
    evaluate_set = evaluator.evaluate()
    print(evaluate_set)

    # メイン処理
    for i in range(generation_loop):

        # 親個体の選択後，index配列取得
        parents_index = RouletteSelector(dimension + 1).select(individual_set, evaluate_set)

        # 交叉
        children_set = Simplex(dimension * 10).crossover(individual_set, parents_index)

        # 選択
        individual_set = JGG().select(individual_set, parents_index, children_set)

        # 最良個体
        evaluator = Evaluator(individual_set)
        evaluate_set = evaluator.evaluate()
        best_value = np.sort(evaluate_set)
        print(i, best_value[0], individual_set[0])

if __name__ == "__main__":
    test()
