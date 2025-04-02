import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from data.load_data import q2_ab_data


def q2_a():

    data = q2_ab_data()

    flatten_data = pd.DataFrame(columns=['timestamp', 'ED_volume', 'ID'], dtype=float)

    for j, _ in data.iterrows():
        for i in range(1, 9):
            a = data.loc[j, f'timestamp.{i}']
            if not pd.isna(a):
                flatten_data = flatten_data._append({
                    'ID': data.loc[j, 'ID'],
                    'timestamp': data.loc[j, f'timestamp.{i}'],
                    'ED_volume': data.loc[j, f'ED_volume.{i}']
                }, ignore_index=True)


    x = flatten_data['timestamp'].to_numpy()
    y = flatten_data['ED_volume'].to_numpy() + 10



    # x = x[y != 0.]
    # y = y[y != 0.]

    sort_idx = np.argsort(x)




    # sns.heatmap(flatten_data[['timestamp', 'ED_volume']].to_numpy(), cmap="YlGnBu", ax=ax)
    # ax.grid()
    # plt.show()

    x = np.log(x)
    y = np.log(y)


    pe = PolynomialFeatures(degree=20)
    xx = pe.fit_transform(x.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(xx, y)
    pre = lr.predict(xx)

    fig, ax_arr = plt.subplots(nrows=1, ncols=2)
    ax = ax_arr[0]
    ax.scatter(x, y)
    ax.plot(x[sort_idx], pre[sort_idx], c='orange')


    bx = np.exp(x)
    by = np.exp(y)
    pre = np.exp(pre)


    ax = ax_arr[1]
    ax.scatter(bx, by)
    ax.plot(bx[sort_idx], pre[sort_idx], c='orange')

    mse = mean_absolute_error(by, pre)
    print(mse)


    plt.show()
    #  TODO 输出结果至表格


def q2_b():
    data = q2_ab_data()
    flatten_data = pd.DataFrame(columns=['timestamp', 'ED_volume', 'ID'], dtype=float)

    for j, _ in data.iterrows():
        for i in range(1, 9):
            a = data.loc[j, f'timestamp.{i}']
            if not pd.isna(a):
                flatten_data = flatten_data._append({
                    'ID': data.loc[j, 'ID'],
                    'timestamp': data.loc[j, f'timestamp.{i}'],
                    'ED_volume': data.loc[j, f'ED_volume.{i}']
                }, ignore_index=True)

    # print(flatten_data.shape)
    which_person_arr = np.zeros(shape=(data.shape[0], flatten_data.shape[0]))

    for i in range(100):
        idx = flatten_data[flatten_data['ID'] == 'sub{:03d}'.format(i + 1)].index.to_numpy()
        which_person_arr[i][idx] = 1

    # print(which_person_arr)

    print(which_person_arr.shape)

    def fit_trans_poly(x, y, degree):

        # 自定义征转换器
        class FeatureTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, np_function=np.log):
                self.np_function = np_function

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # 添加sin特征
                feature = self.np_function(X)
                # 返回包含sin特征的新数组
                # return np.hstack([X, sin_feature])
                return feature

        steps = [
            ('log', FeatureTransformer(np.log)),
            ('sin', FeatureTransformer(np.sin)),
            # ('sin', FeatureTransformer(np.cos)),
            # ('norm', Normalizer()),
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ]
        pipeline = Pipeline(steps)

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        pipeline.fit(x, y)
        return pipeline.predict(x), pipeline

    def cal_which_group(which_person, maes_, ):
        mae_person_sum = np.dot(which_person, maes_)
        mae_person_mean = mae_person_sum / which_person.sum(axis=1).reshape(-1, 1)
        return mae_person_mean


    x = flatten_data['timestamp'].to_numpy()
    y = flatten_data['ED_volume'].to_numpy()
    degrees = [5, 7, 12, 18, 20]

    # 记录每个点在每个曲线的残差
    maes = np.zeros(shape=(flatten_data.shape[0], len(degrees)))

    # 记录曲线平均误差
    mae_curve = np.zeros(shape=(len(degrees)))

    for i, de in enumerate(degrees):
        pre, model = fit_trans_poly(x, y, de)
        mae = mean_absolute_error(y, pre)
        print(de, mae)
        mae_curve[i] = mae
        maes[:, i] = np.abs(y - pre)

    mae_mean = cal_which_group(which_person_arr, maes)
    min_idx = np.argmin(mae_mean, axis=1)
    print(min_idx)



    best_run = (0, mae_mean.min(axis=1).mean())
    step = 0
    while True:
        print("Step: {}".format(step))
        if step == 17:
            font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
            plt.rcParams['font.family'] = font.get_name()
            # plt.rcParams['font.size'] = 20
            fig, ax_arr = plt.subplots(ncols=len(degrees))

        last_mae_curve = mae_curve.copy()

        # 用于记录每组有多少点
        curve_point_num = np.zeros(shape=(len(degrees)))

        for model_num, de in enumerate(degrees):
            which_data = which_person_arr[min_idx == model_num].sum(axis=0).astype(bool)
            # 如果该曲线没有分到点
            if not which_data.sum():
                continue
            # print(which_data)
            pre, model = fit_trans_poly(x[which_data], y[which_data], de)
            mae = mean_absolute_error(y[which_data], pre)

            print(de, mae, which_data.sum())
            mae_curve[model_num] = mae
            curve_point_num[model_num] = which_data.sum()

            maes[:, model_num] = np.abs(y - model.predict(x.reshape(-1, 1)))
            # print(pre)
            if step == 17:

                sl = []
                lr = model['linear']
                print(lr.intercept_)
                print(lr.coef_)
                ys = 'y'
                xs = ['\\frac{', '}{log', 'x}']
                for i, j in enumerate([lr.intercept_] + lr.coef_[1:].tolist()):
                    if 1000 > j > 0.001 or -1000 < j < -0.001:
                        ts = "{:.2f}".format(j)
                    else:
                        ts = "{:.2e}".format(j)
                        ts = ts.split('e')
                        ts[1] = str(int(ts[1]))
                        ts = ts[0] + "\\times 10^{" + ts[1] + '}'
                    # if i == 0:
                    #     ts = ts
                    # elif i == 1:
                    #     ts = xs[0] + ts + xs[1] + xs[2]
                    # else:
                    #     ts = xs[0] + ts + xs[1] + "^{" + str(i) + "}" + xs[2]
                    sl.append(ts)
                # s = "$" + ys + '=' + "+".join(sl[:]) + '$'
                # print(s.replace('+-', '-'))
                print("~".join(sl) + r'\\')
                ax = ax_arr[model_num]
                sorted_idx = x[which_data].argsort()
                ax.scatter(x[which_data][sorted_idx], y[which_data][sorted_idx])
                # ax.plot(x[which_data][sorted_idx], pre[sorted_idx], c='orange')
                ax_x = np.linspace(start=x[which_data].min(), stop=x[which_data].max(), num=1000)
                ax.plot(ax_x, model.predict(ax_x.reshape(-1, 1)), c='orange')
                ax.set_xlabel('发病时间(s)')
                ax.set_ylabel('水肿体积($10_{-3}$ml)')

        if step == 17:
            plt.show()

        mae_mean = cal_which_group(which_person_arr, maes)
        min_idx = np.argmin(mae_mean, axis=1)

        if best_run[1] > mae_mean.min(axis=1).mean():
            best_run = (step, mae_mean.min(axis=1).mean())
        print(best_run)
        print((step, mae_mean.min(axis=1).mean()))
        print()
        if step == 17:
            exit()

        if (mae_curve == last_mae_curve).all():
            bad_curve = np.argmax(mae_curve)
            bad_idx_bool = min_idx == bad_curve
            # print(bad_idx_bool)

            # min_idx[bad_idx_bool] = list(range(len(degrees))) * (bad_idx_bool.sum() // len(degrees)) + list(range(bad_idx_bool.sum() % len(degrees)))

            if curve_point_num.min() == curve_point_num[bad_curve]:
                degrees[bad_curve] = min(degrees) - 1
            elif curve_point_num.max() == curve_point_num[bad_curve]:
                degrees[bad_curve] = max(degrees) + 1
            else:
                # min_idx[bad_idx_bool] = list(range(len(degrees))) * (bad_idx_bool.sum() // len(degrees)) + list(range(bad_idx_bool.sum() % len(degrees)))
                min_idx[bad_idx_bool] = list(range(len(degrees))) * (bad_idx_bool.sum() // len(degrees)) + list(range(bad_idx_bool.sum() % len(degrees)))

        # time.sleep(0.2)
        step += 1
        # break





if __name__ == "__main__":
    # q2_a()
    q2_b()
