# メインファイル

from generator import *
from evaluator import *
from crossover import *
from individual_selector import *
from generation_selector import *
import numpy as np
from matplotlib import pyplot as plt

def test():
    # 遺伝子の値の最大値
    maximum = 10

    # 遺伝子の値の最小値
    minimum = 0

    # パラメータ数
    dimension = 2

    #　個体数
    size = 300

    # 繰り返し数
    generation_loop = 100


    # テストデータを取得
    data = StandardData(5, 2).standard("result.csv")

    # 評価関数
    evaluator = CrossValidation()


    # グラフ化用の配列
    count = np.array([], dtype = np.int)
    value = np.array([], dtype = np.float)

    # 初期個体の生成
    generator = Generator(maximum, minimum, dimension, size)
    individual_set = generator.generate()

    # 初期個体の評価
    evaluate_set = evaluator.evaluate(individual_set, data)

    # メイン処理
    for i in range(generation_loop):

        # 親個体の選択後，index配列取得
        parents_index = RouletteSelector(dimension + 1).select(individual_set, evaluate_set)

        # 交叉
        children_set = Simplex(dimension * 10).crossover(individual_set, parents_index)
        children_value = evaluator.evaluate(children_set, data)

        # 選択
        individual_set = JGG().select(individual_set, parents_index, children_set, children_value)

        # 最良個体
        evaluate_set = evaluator.evaluate(individual_set, data)
        best_value = np.sort(evaluate_set)
        print(i, best_value[0], individual_set[0])

        # 評価値の履歴
        count = np.append(count, i)
        value = np.append(value, best_value[0])

    # 評価値の履歴をグラフ化
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # グリッド
    ax.grid(zorder=0)

    # ラベルの指定
    ax.set_xlabel(r'count')
    ax.set_ylabel(r'value')
    ax.set_xlim(0.0, generation_loop)

    ax.scatter(count, value, s=10, c='blue', edgecolors='blue', linewidths='1', marker='o', alpha = '0.5')
    plt.show()

if __name__ == "__main__":
    test()
