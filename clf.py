from sklearn.datasets import fetch_openml
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784", version=1, cache=True)
# help(fetch_openml)
# print(mnist.data)
# print(mnist.target, end="\n\n")
# print(mnist.details, end="\n\n")
# print(mnist.DESCR)


def plot_data(data):
    images = np.array(data).reshape(28, 28)
    plt.imshow(images, cmap="binary", interpolation="nearest")
    plt.savefig("pic.png")
    plt.show()


x_train, x_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], test_size=0.3)
# x, y = mnist["data"], mnist["target"]
# print(x.shape, type(x))
# plot_data(x.loc[36000])
# print(x[2:4])
print(x_train.shape, x_test.shape, y_test.shape, y_train.shape)
# 训练一个分类器
from sklearn.linear_model import SGDClassifier

print(y_train, y_test)
y_train5 = y_train == "5"
y_test5 = y_test == "5"
# print(type(y_train[367]), y_train[367])
print(np.unique(y_train5), np.unique(y_test5), end="\n\n")
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(x_train, y_train5)
print(sgd_clf.predict(x_test[1:10]), y_test5[1:10], sep="\n")

# 分层交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
X_train, y_train_5 = np.array(x_train), np.array(y_train5)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# 多标签分类器
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (np.array(y_train, dtype=np.int32)) >= 7
y_train_odd = (np.array(y_train, dtype=np.int32)) % 2 == 1
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
print(y_multilabel)
knn_clf.fit(x_train, y_multilabel)
knn_clf.predict(x_test[1:10])
