# メインファイル
from generator import *
from evaluator import *
from crossover import *
from individual_selector import *
from generation_selector import *
import numpy as np
import time
from matplotlib import pyplot as plt

t1 = time.time()

def main():
    # GAパラメータ
    maximum = 0.2            # 遺伝子の値の最大値
    minimum = 0              # 遺伝子の値の最小値
    dimension = 2            # パラメータ数
    size = 100               # 個体数
    generation_loop = 1000   # 繰り返し数

    # 訓練用データ
    design = 5              # 設計変数の数
    object = 2              # 目的関数の数
    num = 0                 # 何番目の目的関数について調べるか
    filename = "result.csv" # 訓練データのcsvファイル

    # 訓練用データの取り込み
    data = StandardData(design, object).standard(filename)

    # 評価関数
    evaluator = CrossValidation(data, design, object, num)
    #evaluator = Rosenbrock()

    # グラフ化用の配列
    count = np.array([], dtype = np.int)
    value = np.array([], dtype = np.float)

    # 初期個体の生成
    generator = Generator(maximum, minimum, dimension, size)
    individual_set = generator.generate()

    # 初期個体の評価
    evaluate_set = evaluator.evaluate(individual_set)

    # メイン処理
    for i in range(generation_loop):

        # 親個体の選択後，index配列取得
        parents_index = RouletteSelector(dimension).select(individual_set, evaluate_set)
        #parents_index = RouletteSelector(dimension + 1).select(individual_set, evaluate_set)

        # 交叉
        children_set = BLXalpha(dimension * 10).crossover(individual_set, parents_index)
        #children_set = Simplex(dimension * 10).crossover(individual_set, parents_index)
        children_value = evaluator.evaluate(children_set)

        # 選択
        individual_set = JGG().select(individual_set, parents_index, children_set, children_value)

        # 最良個体
        evaluate_set = evaluator.evaluate(individual_set)
        best_value = np.sort(evaluate_set)
        print(i, best_value[0], individual_set[0])

        # 評価値の履歴
        count = np.append(count, i)
        value = np.append(value, best_value[0])

    # 計算時間の表示
    t2 = time.time()
    elapsed_time = t2 - t1
    print(elapsed_time)

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
    main()
