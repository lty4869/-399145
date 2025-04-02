import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def add_NCCT_original_firstorder_Skewness(data: pd.DataFrame):
    x = data['NCCT_original_firstorder_Skewness']
    print(x.shape)
    c1 = data['扩张比例']

    p = PolynomialFeatures(degree=2)
    x = x.to_numpy()
    print(data.shape)

    idx = np.argsort(x[:100])
    y_idx = np.where((c1[:100].to_numpy() > 0.33) & (x[:100] < 0.5))[0]
    xxx = p.fit_transform(x[:100].reshape(-1, 1))

    lr = LinearRegression()
    print(y_idx)

    lr.fit(np.concatenate([xxx] + 4 * [xxx[y_idx]], axis=0),
           np.concatenate([c1[:100]] + 4 * [c1[y_idx]], axis=0)
           )

    pre = lr.predict(xxx)
    print(p.powers_)
    print(max(c1))
    print(max(x[:100]), min(x[100:]), max(x[100:]))
    # exit(0)

    # # fig, ax = plt.subplots()
    plt.scatter(x[:100], c1[:100], c=['pink' if i > 0.33 else 'blue' for i in c1][:100])
    plt.scatter(x[100:], c1[100:], c=['red' if i > 0.33 else 'green' for i in c1][100:])
    # # ax.set_ylabel('y')
    # # ax.set_xlabel('x')
    # # ax.spines['right'].set_visible(False)  # ax右轴隐藏
    #
    # # plt.show()
    # # z_ax = ax.twinx()  # 创建与轴群ax共享x轴的轴群z_ax
    # # z_ax.scatter(x, c2, c=['red' if i > 6000. else 'black' for i in c2])
    plt.plot(x[:100][idx], pre[idx], c='black')
    plt.plot([-0.5, 1], [0.33, 0.33])
    # # z_ax.set_ylabel('y')

    plt.show()
    print(x.shape)
    return lr.predict(p.transform(x.reshape(-1, 1)))
