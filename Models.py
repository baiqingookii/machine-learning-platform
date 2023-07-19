from collections import defaultdict
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import operator
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from xgboost import XGBRegressor


class Algorithm():
    def predict(self, x_train, y_train):
        pass

    def fit(self, x_test):
        pass


##################################################################################
# 线性回归
class Linear(Algorithm):  # 线性回归
    def __init__(self):
        self.intercept = None
        self.coef = None

    def fit(self, x_train, y_train):
        # x_b为x_train前加一列1
        x_b = np.hstack((np.ones((len(x_train), 1)), x_train))
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]
        return self

    def predict(self, x_test):
        x_b = np.hstack((np.ones((len(x_test), 1)), x_test))
        y_predict = x_b.dot(self._theta)
        return y_predict

    # 计算R 方差
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        mean = np.mean(y_test)
        # mean_squared_error = np.sum((y_test - y_predict) ** 2) / len(y_test)
        # r2_score = 1 - mean_squared_error / np.var(y_test)
        # print(np.sum((y_test - y_predict) ** 2))
        # print(np.sum((y_test - mean) ** 2))
        r2_score = 1 - np.sum((y_test - y_predict) ** 2) / np.sum((y_test - mean) ** 2)
        return r2_score


##################################################################################
# 支持向量机
class SVM(Algorithm):
    def __init__(self, kernel='rbf', degree=3, tol=0.001):
        self.y_train = None
        self.x_train = None
        self.kernel = kernel
        self.degree = degree
        self.tol = tol
        self.model = SVC(kernel=self.kernel, degree=self.degree, tol=self.tol)
        # kernel:核函数类型，默认为‘rbf’，高斯核函数，exp(-gamma|u-v|^2)
        # 其他可选项有：
        # 'linear':线性核函数，u'*v
        # 'poly':多项式核函数，(gamma*u'*v + coef0)^degree
        # 'sigmoid':sigmoid核函数，tanh(gamma*u'*v + coef0)
        # degree:多项式核的阶数，默认为3
        # 对其他核函数不起作用
        # tol:停止训练的误差精度，默认值为0.001

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred


##################################################################################
# K-近邻
class KNN(Algorithm):  # K近邻
    def __init__(self, k=5):
        self.classes = None
        self.k = k
        self.data = None
        self.labels = None
        self.ndim = 0

    def fit(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.ndim = len(self.data[0])

    def predict(self, data, features=None):
        data = np.array(data)
        if features is None:
            features = np.ones(self.data.shape[1])
        else:
            features = np.array(features)

        if data.ndim == 1:
            dist = self.data - data
        elif data.ndim == 2:
            dist = np.zeros((data.shape[0],) + self.data.shape)
            for i, d in enumerate(data):
                dist[i, :, :] = self.data - d
        else:
            raise ValueError("Cannot process data with dimensionality > 2")
        dist = features * dist
        dist = dist * dist
        dist = np.sum(dist, -1)
        dist = np.sqrt(dist)
        nns = np.argsort(dist)

        if data.ndim == 1:
            classes = dict((cls, 0) for cls in self.classes)
            for n in nns[:self.k]:
                classes[self.labels[n]] += 1
            labels = sorted(classes.items(), key=operator.itemgetter(1))[-1][0]
        elif data.ndim == 2:
            labels = list()
            for i, d in enumerate(data):
                classes = dict((cls, 0) for cls in self.classes)
                for n in nns[i, :self.k]:
                    classes[self.labels[n]] += 1
                labels.append(sorted(classes.items(), key=operator.itemgetter(1))[-1][0])

        return labels


##################################################################################
# 逻辑回归
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Logistic(Algorithm):
    def __init__(self, learning_rate=0.003, iterations=100):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate  # 学习率
        self.iterations = iterations  # 迭代次数

    def fit(self, x_train, y_train):
        # 初始化参数
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.weights = np.random.randn(x_train.shape[1])
        self.bias = 0

        # 梯度下降
        for i in range(self.iterations):
            # 计算sigmoid函数的预测值, y_hat = w * x + b
            y_hat = sigmoid(np.dot(x_train, self.weights) + self.bias)

            # 计算损失函数
            loss = (-1 / len(x_train)) * np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))

            # 计算梯度
            dw = (1 / len(x_train)) * np.dot(x_train.T, (y_hat - y_train))
            db = (1 / len(x_train)) * np.sum(y_hat - y_train)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 打印损失函数值
            '''if i % 10 == 0:
                print(f"Loss after iteration {i}: {loss}")'''

    # 预测
    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat


##################################################################################
# 决策树分类
# 结点
class TreeNode(object):
    def __init__(self, index, split_value, feature_type, criterion):
        self.index = index
        self.split_value = split_value
        self.feature_type = feature_type
        self.criterion = criterion

        self.left_child = None
        self.right_child = None

        # if the node is a leaf node, set the label to the category that has
        # appeared the most times
        self.label = None

    def set_left_child(self, node):
        self.left_child = node

    def set_right_child(self, node):
        self.right_child = node

    def split(self, X):
        if self.feature_type == 'num':  # use '<=' to split numeric feature
            left_mask = X.iloc[:, self.index] <= self.split_value
        else:  # use '==' to split string feature
            left_mask = X.iloc[:, self.index] == self.split_value

        return left_mask.values

    def get_impurity(self, X, y):
        if len(X) == 0:
            return 0
        left_mask = self.split(X)
        y_left = y[left_mask]
        y_right = y[~left_mask]
        return getattr(self, self.criterion)(y, y_left, y_right)

    @staticmethod
    def gini(y, y_left, y_right):
        if len(y) == 0:
            return 0
        prob_parent = np.unique(y, return_counts=True)[1] / len(y)
        gini_parent = 1 - np.sum(np.square(prob_parent))

        for y_child in [y_left, y_right]:
            if len(y_child) == 0:
                continue
            prob_child = np.unique(y_child, return_counts=True)[1] / len(y_child)
            gini_child = float(len(y_child)) / float(len(y)) * (1 - np.sum(np.square(prob_child)))
            gini_parent -= gini_child
        return gini_parent

    @staticmethod
    def entropy(y, y_left, y_right):
        raise NotImplementedError

    @staticmethod
    def error(y, y_left, y_right):
        raise NotImplementedError


# 决策树分类主体
class DecisionTreeClassifier(Algorithm):
    def __init__(self, criterion='gini', max_depth=5, d=4,
                 random_state=0):
        self.label_dtype = None
        assert criterion in ['gini', 'entropy',
                             'error'], 'Expect criterion is one of "gini", ' \
                                       '"entropy" or "error", bug got %s' % criterion
        self.criterion = criterion
        self.max_depth = max_depth
        self.d = d
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.squeeze(y.values)
        self.label_dtype = y.dtype

        if len(y.shape) >= 2:
            raise Exception(
                'Expect y has 1d dimension, bug got y.shape=' + str(
                    y.shape))

        milestones = []
        feature_types = []
        for col in X.columns:
            unique_features = np.sort(X[col].unique())
            milestones.append(unique_features)
            if type(X[col].dtype) == 'object':  # str features
                feature_types.append('str')
            else:
                feature_types.append('num')
        milestones = np.asarray(milestones)
        feature_types = np.asarray(feature_types)
        self.tree = self._build_tree(X, y, milestones, feature_types, 0)

    def _build_tree(self, X, y, milestones, feature_types, depth=0):
        assert len(X) == len(y)
        rgen = np.random.RandomState(self.random_state)

        if len(X) == 0 or (
                self.max_depth is not None and depth > self.max_depth):
            return None

        if np.all(y[0] == y):  # the whole data have a same category.
            node = TreeNode(None, None, None, self.criterion)
            node.label = y[0]
            return node

        best_node = None
        best_impurity = 0
        feature_indices = np.arange(len(feature_types))
        sampled_feature_types = feature_types
        sampled_milestones = milestones
        if self.d is not None:
            assert self.d <= len(feature_indices), 'Expect d <= X.shape[1], ' \
                                                   'but got d = %d' % self.d

            feature_indices = rgen.choice(feature_indices, self.d,
                                          replace=False)
            sampled_feature_types = feature_types[feature_indices]
            sampled_milestones = milestones[feature_indices]

        for findex, ftype, frange in zip(feature_indices,
                                         sampled_feature_types,
                                         sampled_milestones):
            for fvalue in frange:
                node = TreeNode(findex, fvalue, ftype, self.criterion)
                impurity = node.get_impurity(X, y)
                if impurity > best_impurity:
                    best_node = node
                    best_impurity = impurity
        if best_node is not None:
            left_mask = best_node.split(X)
            right_mask = ~left_mask
            X_left, y_left = X.loc[left_mask], y[left_mask]
            left_child = self._build_tree(X_left, y_left, milestones,
                                          feature_types, depth + 1)
            X_right, y_right = X.loc[right_mask], y[right_mask]
            right_child = self._build_tree(X_right, y_right, milestones,
                                           feature_types, depth + 1)
            if left_child is None and right_child is None:
                # current node is a leaf node
                catorgeris, counts = np.unique(y, return_counts=True)
                best_node.label = catorgeris[np.argmax(counts)]
            else:
                # Either there are no child nodes or there are two child nodes
                assert left_child is not None and right_child is not None

                best_node.set_left_child(left_child)
                best_node.set_right_child(right_child)
        else:
            best_node = TreeNode(None, None, None, self.criterion)
            catorgeris, counts = np.unique(y, return_counts=True)
            best_node.label = catorgeris[np.argmax(counts)]

        return best_node

    def _recursive_predict(self, X, node):

        if len(X) == 0:
            return np.asarray([], dtype=self.label_dtype)

        if node.label is not None:
            return np.asarray(len(X) * [node.label], dtype=self.label_dtype)

        left_mask = node.split(X)
        left_predict = self._recursive_predict(X.loc[left_mask],
                                               node.left_child)
        right_predict = self._recursive_predict(X.loc[~left_mask],
                                                node.right_child)
        predict = np.empty(shape=len(X), dtype=self.label_dtype)
        predict[left_mask] = left_predict
        predict[~left_mask] = right_predict

        return predict

    def predict(self, X):
        X = np.array(X)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return self._recursive_predict(X, self.tree)


##################################################################################
# K平均
class KMeans(Algorithm):
    def __init__(self, n_clusters=5, initCent='random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None
        self.sse = None

        # 计算两点的欧式距离

    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # 随机选取k个质心,必须在数据集的边界内
    def _randCent(self, X, k):
        n = X.shape[1]  # 特征维数
        centroids = np.empty((k, n))  # k*n的矩阵，用于存储质心
        for j in range(n):  # 产生k个质心，一维一维地随机初始化
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, X):
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # m代表样本数量
        self.clusterAssment = np.empty((m, 2))  # m*2的矩阵，第一列存储样本点所属的族的索引值，
        # 第二列存储该点与所属族的质心的平方误差
        if self.initCent == 'random':
            self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
                minDist = np.inf;
                minIndex = -1
                for j in range(self.n_clusters):
                    distJI = self._distEclud(self.centroids[j, :], X[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j
                if self.clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i, :] = minIndex, minDist ** 2

            if not clusterChanged:  # 若所有样本点所属的族都不改变,则已收敛，结束迭代
                break
            for i in range(self.n_clusters):  # 更新质心，即将每个族中的点的均值作为质心
                ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]  # 取出属于第i个族的所有点
                self.centroids[i, :] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])

    def predict(self, X):  # 根据聚类结果，预测新输入数据所属的族
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # m代表样本数量
        preds = np.empty((m,))
        for i in range(m):  # 将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


##################################################################################
# 随机森林
class RandomForestClassifier(Algorithm):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 d=None,
                 random_state=0):
        self.estimators = [DecisionTreeClassifier(criterion, max_depth, d,
                                                  random_state + i) for i in
                           range(n_estimators)]
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        rgen = np.random.RandomState(self.random_state)
        N, _ = X.shape
        indices = np.arange(N)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.squeeze(y.values)
        for estimator in self.estimators:
            sampled_indices = rgen.choice(indices, size=N, replace=True)
            sampled_X = X.iloc[sampled_indices]
            sampled_y = y[sampled_indices]
            estimator.fit(sampled_X, sampled_y)

    def predict(self, X):
        X = np.array(X)
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X))
        preds = np.asarray(preds)
        preds = np.split(preds, preds.shape[1], axis=1)
        predict = []
        for p in preds:
            label, count = np.unique(p.squeeze(), return_counts=True)
            predict.append(label[np.argmax(count)])
        predict = np.asarray(predict)
        return predict


##################################################################################
# 朴素贝叶斯
class Beyes(Algorithm):
    def __init__(self):
        self.length = -1  # 保存测试集数据量
        self.train_target_list = []  # 目标值类别集合
        self.p_train_target = {}  # 保存个目标值概率
        self.split_data_lis = []  # 保存各条件概率对应的数据集
        self.feature_p_lis = []  # 保存特征概率
        self.y_predict = []  # 保存分类结果

    def fit(self, train_data, train_target):
        train_data = np.array(train_data)
        train_target = np.array(train_target)
        train_length = train_data.shape[0]
        self.length = train_length
        target_list = list(set(train_target))  # 队训练集目标值去重
        self.train_target_list = target_list  # 写入对象特征
        target_classifier = dict(Counter(train_target))  # 保存目标值的分类计数（字典格式）
        train_data = pd.DataFrame(train_data)
        train_data['target'] = train_target  # 将数据转换为DataFrame格式方便后续聚合
        for target in self.train_target_list:
            self.p_train_target[target] = target_classifier[target] / self.length  # 保存各目标值的概率
            split_data = train_data[train_data['target'] == target]
            self.split_data_lis.append(split_data)
        print('model had trained please use classifier() to get result')

    def p_test_data(self, sample):
        result_p = []
        for j in range(len(self.train_target_list)):
            p_label = 1
            this_target = self.train_target_list[j]
            this_data = self.split_data_lis[j]
            for i in range(0, sample.shape[0]):
                feature_num_dict = dict(Counter(this_data[i]))  # 计算一列数据中各类别的数量
                if sample[i] in feature_num_dict:
                    label_num = feature_num_dict.get(sample[i])
                    p_label = p_label * (label_num / this_data.shape[0])  # 计算单个特征的条件概率
                else:
                    # 加入拉普拉斯平滑系数解决概率为0的情况'
                    p_label = p_label * (1 / (this_data.shape[0] + len(feature_num_dict)))
            this_target_p = p_label * self.p_train_target.get(this_target)  # 计算该样本属于该特征的概率
            result_p.append(this_target_p)
        position = result_p.index(max(result_p))  # 概率最大的分类
        return self.train_target_list[position]

    def predict(self, test_data):
        test_data = np.array(test_data)
        if self.length == -1:
            raise ValueError('please use fit() to train the train data set ')
        else:
            test_data = pd.DataFrame(test_data)
            test_data['target'] = test_data.apply(self.p_test_data, axis=1)  #
            self.y_predict = list(test_data['target'])
            return self.y_predict


##################################################################################
# 降维算法
# 降维算法主要用于数据预处理
class DimReduction(Algorithm):
    def __init__(self, n_components=None, whiten=False):
        self.x_train = None
        self.y_train = None
        self.n_components = n_components  # 返回所保留的成分个数n。
        self.whiten = whiten  # 白化，使得每个特征具有相同的方差。

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test):  # 降维算法里面这是转化
        x_test = np.array(x_test)
        model = PCA(n_components=self.n_components, whiten=self.whiten)
        model.fit_transform(self.x_train)
        x_pred = model.transform(x_test)
        return x_pred[0:5]


##################################################################################
# 梯度增强
# 这里使用XGboost
class Graboosting(Algorithm):
    def __init__(self, flag=0, eta=0.3, max_depth=6, subsample=1):
        self.flag = flag  # 0默认为分类，1为回归
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        if self.flag == 0:
            model = XGBClassifier(eta=self.eta, max_depth=self.max_depth, subsample=self.subsample)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(x_test)
            return y_pred
        if self.flag == 1:
            model = XGBRegressor(max_depth=self.max_depth,  # 每一棵树最大深度，默认6；
                                 learning_rate=self.eta,  # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                                 n_estimators=100,  # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                                 objective='reg:squarederror',  # 此默认参数与 XGBClassifier 不同，‘
                                 booster='gbtree',
                                 # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                                 gamma=0.2,  # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                                 min_child_weight=7,  # 可以理解为叶子节点最小样本数，默认1；
                                 subsample=self.subsample,  # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                                 colsample_bytree=0.8,  # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                                 reg_alpha=0.01,  # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                                 reg_lambda=1,  # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                 nthread=4,
                                 scale_pos_weight=2,
                                 seed=27
                                 )
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(x_test)
            return y_pred


##################################################################################
class Tree_model:
    def __init__(self, stump, mse, left_value, right_value, residual):
        '''
        :param stump: 为feature最佳切割点
        :param mse: 为每棵树的平方误差
        :param left_value: 为决策树左值
        :param right_value: 为决策树右值
        :param residual: 为每棵决策树生成后余下的残差
        '''
        self.stump = stump
        self.mse = mse
        self.left_value = left_value
        self.right_value = right_value
        self.residual = residual


'''根据feature准备好切分点。例如:
feature为[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
切分点为[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
'''


class GBDT(Algorithm):
    def __init__(self, criterion='gini', max_depth=5, d=4,
                 random_state=0):
        assert criterion in ['gini', 'entropy',
                             'error'], 'Expect criterion is one of "gini", ' \
                                       '"entropy" or "error", bug got %s' % criterion
        self.criterion = criterion
        self.max_depth = max_depth
        self.d = d
        self.random_state = random_state

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.tree = DecisionTreeClassifier(self.criterion, self.max_depth, self.d, self.random_state)
        self.tree.fit(x_train, y_train)
        y_predict0 = self.tree.predict(self.x_train)
        self.feature = y_predict0
        self.label = y_train
        # self.Trees = self.BDT_model(y_predict0, self.y_train)
        self.Trees = self.BDT_model()
        # print(1)

    def predict(self, x_test):
        self.x_test = np.array(x_test)
        # y_predict0 = self.tree.predict(self.x_train)
        y_predict = self.tree.predict(self.x_test)
        y_predict = np.array(y_predict)
        predict = self.BDT_predict(self.Trees, y_predict)
        # predict=0
        num = len(predict)
        predict_copy = predict.copy()
        for i in range(num):
            if predict[i] > 0.5:
                predict_copy[i] = 1
            else:
                predict_copy[i] = 0

        return predict_copy

    def Get_stump_list(self):
        # 特征值从小到大排序好,错位相加
        tmp1 = list(self.feature.copy())
        tmp2 = list(self.feature.copy())
        tmp1.insert(0, 0)
        tmp2.append(0)
        # stump_list = ((np.array(tmp1) + np.array(tmp2)) / 2.0)[1:-1]
        stump_list = ((np.array(tmp1) + np.array(tmp2)) / 2.0)
        return stump_list

    # 此处的label其实是残差
    def Get_decision_tree(self, stump_list, feature, label):
        best_mse = np.inf
        best_stump = 0  # min(stump_list)
        residual = np.array([])
        left_value = 0
        right_value = 0
        for i in range(np.shape(stump_list)[0]):
            left_node = []
            right_node = []
            for j in range(np.shape(feature)[0]):
                if feature[j] < stump_list[i]:
                    left_node.append(label[j])
                else:
                    right_node.append(label[j])
            left_mse = np.sum((np.average(left_node) - np.array(left_node)) ** 2)
            right_mse = np.sum((np.average(right_node) - np.array(right_node)) ** 2)
            # print("decision stump: %d, left_mse: %f, right_mse: %f, mse: %f" % (i, left_mse, right_mse, (left_mse + right_mse)))
            if best_mse > (left_mse + right_mse):
                best_mse = left_mse + right_mse
                left_value = np.average(left_node)
                right_value = np.average(right_node)
                best_stump = stump_list[i]
                left_residual = np.array(left_node) - left_value
                right_residual = np.array(right_node) - right_value
                residual = np.append(left_residual, right_residual)
                # print("decision stump: %d, residual: %s"% (i, residual))
        Tree = Tree_model(best_stump, best_mse, left_value, right_value, residual)
        return Tree, residual

    # Tree_num就是树的数量
    def BDT_model(self, Tree_num=100):
        self.feature = np.array(self.feature)
        self.label = np.array(self.label)
        stump_list = self.Get_stump_list()
        Trees = []
        residual = self.label.copy()
        # 产生每一棵树
        for num in range(Tree_num):
            # 每次新生成树后，还需要再次更新残差residual
            Tree, residual = self.Get_decision_tree(stump_list, self.feature, residual)
            # Tree, residual = self.Get_decision_tree(stump_list, feature, residual)
            Trees.append(Tree)
        return Trees

    def BDT_predict(self, Trees, feature):
        predict_list = [0 for i in range(np.shape(feature)[0])]
        # 将每棵树对各个特征预测出来的结果进行相加，相加的最后结果就是最后的预测值
        for Tree in Trees:
            for i in range(np.shape(feature)[0]):
                if feature[i] < Tree.stump:
                    predict_list[i] = predict_list[i] + Tree.left_value
                else:
                    predict_list[i] = predict_list[i] + Tree.right_value
        return predict_list

    ##################################################################################


import time


def print_progress_bar(total_time=0):
    list_circle = ["\\", "|", "/", "—"]
    for i in range(total_time * 4):
        time.sleep(0.25)
        print("\r{}".format(list_circle[i % 4]), end="", flush=True)


if __name__ == '__main__':
    # 生成一些随机数据，用于测试
    np.random.seed(42)
    X = np.random.uniform(0, 10, size=(100, 2))
    # print(X)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, size=100)
    # print(y)
    #
    print_progress_bar(200)
    # df = pd.read_csv('DataSets/BostonHousing.csv')
    # df1 = df.copy()
    #
    # y = df1['medv']
    # del df1['medv']
    # x = df1
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.99)
    #
    # # print(x_train)
    # print(y_test)
    #
    # model = Linear()
    #
    # # model = DimReduction(n_components=3, whiten=True)
    #
    # model.fit(x_train, y_train)
    #
    # y_pred = model.predict(x_test)
    # print(y_pred)
    # print(model.score(x_test, y_test))
