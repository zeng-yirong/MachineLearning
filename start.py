from cProfile import label
import os
import urllib.request
from matplotlib.pyplot import colorbar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def setseed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def class_prop(data):
    return data["income_cat"].value_counts() / len(data)


# print(os.path.abspath("."))
csv_path = os.path.join(os.path.abspath(""), "housing.csv")
housing = pd.read_csv(csv_path)
# 类别划分
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=range(1, 6))
# 随机划分
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# 分层划分
xtrain, xtest = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing["income_cat"])
# print(housing.head())
res = pd.DataFrame({"overall": class_prop(housing), "random": class_prop(test_set), "stratified": class_prop(xtest)})
print(res)
housing.drop("income_cat", axis=1, inplace=True)
housing = xtrain.copy()
# ax = housing.plot(kind="scatter",x="longitude",y="latitude",title="distribute",alpha=0.1)
ax = housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    title="distribute",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    c="median_house_value",
    cmap="jet",
    colorbar=True,
    sharex=False,
)

fig = ax.get_figure()
fig.savefig("res.png")

# 缺失属性的数据
incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(incomplete_rows)

