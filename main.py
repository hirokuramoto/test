# メインファイル
import numpy as np
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

from generator import *
from crossover import *
from individual_selector import *
from generation_selector import *
from userFunction import evaluator

def main():
    '''GAのメイン処理'''

    t1 = time.time()

    # GAパラメータ
    boundary  = './userFunction/bound_test.csv'
    DIMENSION = 99    # パラメータ数
    SIZE      = 100   # 生成する個体数
    LOOP      = 10000 # 繰り返し数
    K         = 1     # 親個体数の調整パラメータ（REXで使用） n+K (K=1～0.5n)

    # 記録用の配列
    count = np.array([], dtype = np.int)
    value = np.array([], dtype = np.float)
    parameter = np.array([], dtype = np.float)

    # 評価関数の設定
    #function = evaluator.CrossValidation()
    function = evaluator.Rosenbrock()
    #function = evaluator.Benchmark()

    # 初期個体の生成
    generator = Generator(boundary, DIMENSION, SIZE)
    individual_set = generator.generate()

    # 初期個体の評価
    evaluate_set = function.evaluate(individual_set)

    # メイン処理
    for i in tqdm(range(LOOP)):

        # 親個体の選択後，index配列取得
        parents_index = RandomSelector(DIMENSION + K).select(individual_set, evaluate_set)

        # 交叉
        #children_set = BLXalpha(DIMENSION * 10).crossover(individual_set, parents_index)
        #children_set = Simplex(DIMENSION * 10).crossover(individual_set, parents_index)
        children_set = REX(DIMENSION * 10, K).crossover(individual_set, parents_index)

        # 子個体の評価
        children_value = function.evaluate(children_set)

        # 選択
        individual_set = JGG().select(individual_set, parents_index, children_set, children_value)

        # 最良個体
        evaluate_set = function.evaluate(individual_set)
        best_value = np.sort(evaluate_set)[0]
        best_individual = individual_set[np.argsort(evaluate_set)[0]]

        # 評価値の履歴
        count = np.append(count, i)
        value = np.append(value, best_value)
        parameter = np.append(parameter, individual_set)

    # 計算結果の表示
    t2 = time.time()
    elapsed_time = t2 - t1

    param2 = parameter.reshape([LOOP*SIZE,-1])
    print("elapsed_time=", elapsed_time)
    print("generation=", i, "best value=", best_value, "parameter=", best_individual)

    #np.savetxt('pop_vars_eval.txt', param2, delimiter='\t', fmt='%.9f')

    # 評価値の履歴をグラフ化
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # グリッド
    ax.grid(zorder=0)

    # グラフラベルの指定
    ax.set_xlabel(r'count')
    ax.set_ylabel(r'value')
    ax.set_xlim(0.0, LOOP)
    ax.scatter(count, value, s=10, c='blue', edgecolors='blue', linewidths=1, marker='o', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
