from random import sample, random, choice, randint
import pandas as pd
import numpy as np
from math import ceil, log
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as reset


# from utils import run_time


class Node(object):
    def __init__(self, size):
        # Node size
        self.size = size
        # Feature to split
        self.split_feature = None
        self.split_point = None
        self.left = None
        self.right = None


class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        self.height = 0
        # In case of n_samples is greater than n
        n = len(X)
        if n_samples > n:
            n_samples = n
        self.root = Node(n_samples)
        self._build_tree(X, n_samples, max_depth)

    def _get_split(self, X, idx, split_feature):

        # The split point should be greater than min(X[feature])
        unique = set(map(lambda i: X[i][split_feature], idx))
        # Cannot split
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        x_min, x_max = min(unique), max(unique)
        # Caution: random() -> x in the interval [0, 1).
        return random() * (x_max - x_min) + x_min

    def _build_tree(self, X, n_samples, max_depth):
        # Dataset shape
        m = len(X[0])
        n = len(X)
        # Randomly selected sample points into the root node of the tree
        idx = sample(range(n), n_samples)
        # Depth, Node and idx
        que = [[0, self.root, idx]]
        # BFS
        while que and que[0][0] <= max_depth:
            depth, nd, idx = que.pop(0)
            # Stop split if X cannot be splitted
            nd.split_feature = choice(range(m))
            nd.split_point = self._get_split(X, idx, nd.split_feature)
            if nd.split_point is None:
                continue
            # Split
            idx_left = []
            idx_right = []
            while idx:
                i = idx.pop()
                xi = X[i][nd.split_feature]
                if xi < nd.split_point:
                    idx_left.append(i)
                else:
                    idx_right.append(i)
            # Generate left and right child
            nd.left = Node(len(idx_left))
            nd.right = Node(len(idx_right))
            # Put the left and child into the que and depth plus one
            que.append([depth + 1, nd.left, idx_left])
            que.append([depth + 1, nd.right, idx_right])
        # Update the height of IsolationTree
        self.height = depth

    def _predict(self, xi):
        # Search xi from the IsolationTree until xi is at an leafnode
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if xi[nd.split_feature] < nd.split_point:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size


class IsolationForest(object):
    def __init__(self):
        self.trees = None
        self.adjustment = None  # TBC

    def fit(self, X, n_samples, max_depth=8, n_trees=100):

        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]

    def _get_adjustment(self, node_size):

        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret

    def _predict(self, xi):
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees
        # Scale
        return 2 ** -(score / self.adjustment)

    def predict(self, X):
        score_list = []
        point_list = []
        for xi in X:
            point_list.append(xi)
            score_list.append(self._predict(xi))
        return point_list, score_list


def points_list(dataset):
    points_list = []
    time_list = []
    f = lambda CO2_value, Tem_value: [float(CO2_value), float(Tem_value)]
    for index, row in dataset.iterrows():
        points_list.append(f(*row[['CO2_value', 'Tem_value']].values.tolist()))
        time_list.append(*row[['time']].values.tolist())
    return points_list, time_list


def train_test_split(data, test_size, shuffle_state=True, random_state=None):
    if shuffle_state:
        data = reset(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test


def whisker(score_list):
    score_list.sort()
    lower_quartile = score_list[int(len(score_list) / 4)]
    upper_quartile = score_list[int(3 * (len(score_list) / 4))]
    quarterback = 2 * (upper_quartile - lower_quartile)
    lower_whisker = lower_quartile - 1.5 * quarterback
    upper_whisker = upper_quartile + 1.5 * quarterback
    #
    return lower_whisker, upper_whisker


def get_outlier(point_list, time_list, score_list, lower_whisker, upper_whisker):
    outlier_list = []
    outlier_time = []
    for i in range(len(score_list)):
        if (score_list[i] < lower_whisker) or (score_list[i] > upper_whisker):
            outlier_list.append(point_list[i])
            outlier_time.append(time_list[i])
    return outlier_list, outlier_time


# @run_time
def main():
    print("Comparing average score of X and outlier's score...")
    data = pd.read_csv('E:\Group project\Test data\\1.003_combine.csv')
    (training, testing) = train_test_split(data, 0.2)
    # #
    points_list_testing, time_list_testing = points_list(training)
    points_list_data, time_list_data = points_list(data)
    points_list_data.append([900, 28])
    time_list_data.append('insert attack time')

    # points_list_testing.append([900, 21])

    # points_list_training = [[random() for _ in range(2)] for _ in range(100)]
    # Add outliers
    # points_list_training.append([2] * 2)
    clf1 = IsolationForest()
    clf1.fit(points_list_testing, n_samples=len(points_list_testing))
    point_list1, score_list1 = clf1.predict(points_list_testing)

    lower_whisker, upper_whisker = whisker(score_list1)
    print('lower whisker: %s, upper whisker: %s' % (lower_whisker, upper_whisker))

    clf2 = IsolationForest()
    clf2.fit(points_list_data, n_samples=len(points_list_data))
    point_list2, score_list2 = clf2.predict(points_list_data)

    outlier_list, outlier_time = get_outlier(point_list2, time_list_data, score_list2, lower_whisker, upper_whisker)

    for i in range(len(outlier_list)):
        print('%s: %s' % (outlier_time[i], outlier_list[i]))

    plt.figure()
    plt.scatter(*zip(*point_list2), c='black')
    if len(outlier_list) != 0:
        plt.scatter(*zip(*outlier_list), c='red')

    plt.xlabel('CO2_value')
    plt.ylabel('Tem_value')
    # plt.axis([0, 3, 0, 3])
    plt.axis([200, 1000, 20, 30])
    plt.show()


if __name__ == "__main__":
    main()
