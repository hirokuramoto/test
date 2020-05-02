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
    def crossover(self, individual_set, parents_index):
        """交叉を実行する
        Args:
            individual_set (np.array) : 個体の2次元配列
            parents_index (np.array)  : 個体集団の中で使用する親個体
        Returns :
            children_set (np.array)   : 生成した子個体
        """
        pass

class Simplex(Crossover):
    def crossover(self, individual_set, parents_index):
        """次元数+1個体から子個体を生成する
        """
        # 親個体の2次元配列化（親個体数×次元数）
        matrix_parents = np.array([individual_set[i] for i in parents_index])
        # 列ごと(axis=0)の平均値を求める（1×次元数）
        center = matrix_parents.mean(axis = 0)
        # 次元数を取得
        dimension = len(center)

        alpha = math.sqrt(dimension + 2)
        # matrix行列の最終行がp^nに相当
        matrix = center + alpha * (matrix_parents - center)
        # 空配列を用意
        children_set = np.array([], dtype = np.float64)

        for _ in range(self._generate_size):
            # 子1個体の遺伝子を格納する空配列を用意
            gene = np.zeros(dimension)
            for k, (vector1, vector2) in enumerate(zip(matrix, matrix[1:])):
                r_k = random.uniform(0., 1.) ** (1./(k+1))
                gene = r_k * (vector1 - vector2 + gene)
            gene += matrix[-1]
            children_set = np.append(children_set, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)
        children_set = np.clip(children_set, 0, None)
        return children_set


class BLXalpha(Crossover):
    def __init__(self, generate_size, alpha = 0.5):
        super(BLXalpha, self).__init__(generate_size)
        self._alpha = alpha

    def crossover(self, individual_set, parents_index):
        """2個体から子個体を生成する
        """
        # 親個体の2次元配列化（親個体数×次元数）
        matrix_parents = np.array([individual_set[i] for i in parents_index])

        # 列ごと(axis=0)の最大・最小を求める（1×次元数）
        gene_max = matrix_parents.max(axis=0)
        gene_min = matrix_parents.min(axis=0)
        gene_abs = np.abs(gene_max - gene_min)

        # 探索範囲を拡大　np.clip(arr, 最小, 最大)で範囲内に収める
        gene_max = np.clip((gene_max + self._alpha * gene_abs), 0, None)
        gene_min = np.clip((gene_min - self._alpha * gene_abs), 0, None)

        # 空配列を用意
        children_set = np.array([], dtype = np.float64)
        for _ in range(self._generate_size):
            gene = [random.uniform(g_min, g_max) for g_max, g_min in zip(gene_max, gene_min)]
            children_set = np.append(children_set, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)
        return children_set


class REX(Crossover):
    def crossover(self, individual_set, parents_index):
        """次元数+1個体から子個体を生成する
        """
        # 親個体の2次元配列化（親個体数×次元数）
        parents = np.array([individual_set[i] for i in parents_index])

        # 親集合の重心を求める.列ごと(axis=0)の平均値を求める（1×次元数)
        center = parents.mean(axis = 0)

        # 次元数を取得
        dimension = len(center)

        # 親個体数を取得
        n = parents.shape[0]

        # 一様分布のときの区間[-a, a]を求める
        a = np.sqrt(3/n)

        # 空配列を用意
        children = np.zeros(dimension)
        children_set = np.array([], dtype = np.float64)

        # 子個体を生成
        for j in range(self._generate_size):
            x = 0.
            for i in range(n):
                x += np.random.uniform(-a, a) * (parents[i, ] - center)
            children = center + x
            children_set = np.append(children_set, children)
        # 2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)
        return children_set

if __name__ == "__main__":
    from generator import *
    from userFunction.evaluator import *
    from individual_selector import *

    generator = Generator(10, 0, 2, 100)
    individual_set = generator.generate()

    evaluator = Rosenbrock()
    evaluate_set = evaluator.evaluate(individual_set)

    random_index = RandomSelector(2+1).select(individual_set, evaluate_set)
    parents = np.array([individual_set[i] for i in random_index])
    test = REX(1000).crossover(individual_set, random_index)
    #print(individual_set)
    print(evaluate_set)
    print(random_index)
    np.savetxt('./individual_set.csv', individual_set, delimiter=',')
    np.savetxt('./children_set.csv', test, delimiter=',')
    np.savetxt('./parents.csv', parents, delimiter=',')
