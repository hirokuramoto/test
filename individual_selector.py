# エリート選択とルーレット選択の定義（親個体のindex配列取得）
from abc import ABCMeta, abstractmethod
import numpy as np
import random

class IndividualSelector(metaclass = ABCMeta):
    '''個体の選択方法のベース'''
    def __init__(self, selection_num):
        self._selection_num = selection_num

    @abstractmethod
    def select(self, individual_set, evaluate_set):
        '''選択した個体の配列を返す
        Args :
            individual_set (np.array) : 個体集団の2次元配列
            evaluate_set (np.array)   : 個体集団の評価値 1次元配列
        Returns :
            selected_index (np.array) : 親集団から選択する個体のindex
        '''
        pass

class RandomSelector(IndividualSelector):
    '''ランダム選択による個体選択'''
    def select(self, individual_set, evaluate_set):
        #個体集団の個数を取得
        n = individual_set.shape[0]
        array = np.arange(n, dtype='int64')

        selected_index = np.random.choice(array, size=self._selection_num, replace=False)
        return selected_index

class EliteSelector(IndividualSelector):
    '''エリート選択による個体選択'''
    def select(self, individual_set, evaluate_set):
        # 昇順ソート後のインデックスを取得
        sort_index = np.argsort(evaluate_set)

        # 必要数分の個体配列のインデックスを返す
        selected_index = np.array(sort_index)[0:self._selection_num]
        return selected_index

class RouletteSelector(IndividualSelector):
    """ルーレット選択による個体選択"""
    def select(self, individual_set, evaluate_set):
        # 評価値が小さいものを選びたい．各評価値から最大評価値を引いて絶対値に直す
        evaluate_abs = np.abs(evaluate_set - np.max(evaluate_set))

        # 要素の和を求める
        total = np.sum(evaluate_abs)

        # 空配列を用意
        selected_index = np.array([], dtype = np.int64)

        for i in range(self._selection_num):
            # 0~totalの範囲のfloat型の乱数を生成
            threshold = random.uniform(0.0, total)
            sum = 0.0
            for index, value in enumerate(evaluate_abs):
                sum  += value
                if sum >= threshold:
                    # 選択されたindexは次の選択から除く．除外indexの配列を作成
                    selected_index = np.append(selected_index, index)

                    # 除外されたindexの評価値を0にする
                    evaluate_abs[index] = 0.
                    total -= value
                    break
        return selected_index

if __name__ == "__main__":
    from generator import *
    from userFunction.evaluator import *

    generator = Generator(10, 0, 3, 4)
    individual_set = generator.generate()
    function = Rosenbrock()
    evaluate_set = function.evaluate(individual_set)

    elite_set = EliteSelector(2).select(individual_set, evaluate_set)
    roulette_set = RouletteSelector(3).select(individual_set, evaluate_set)
    random_set = RandomSelector(3).select(individual_set, evaluate_set)

    print(individual_set)
    print(evaluate_set)
    print(elite_set)
    print(roulette_set)
    print(random_set)
