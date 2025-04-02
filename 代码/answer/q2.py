import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.load_data import q2_ab_data, q2_c_data, q2_d_data


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

    which_person_arr = np.zeros(shape=(data.shape[0], flatten_data.shape[0]))
    for i in range(100):
        idx = flatten_data[flatten_data['ID'] == 'sub{:03d}'.format(i + 1)].index.to_numpy()
        which_person_arr[i][idx] = 1



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


    pe = PolynomialFeatures(degree=7)
    xx = pe.fit_transform(x.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(xx, y)
    pre = lr.predict(xx)
    print(lr.coef_)
    print(lr.intercept_)

    sl = []
    print(lr.intercept_)
    print(lr.coef_)
    for i, j in enumerate([lr.intercept_] + lr.coef_[1:].tolist()):
        if 1000 > j > 0.001 or -1000 < j < -0.001:
            ts = "{:.2f}".format(j)
        else:
            ts = "{:.2e}".format(j)
            ts = ts.split('e')
            ts[1] = str(int(ts[1]))
            ts = ts[0] + "\\times 10^{" + ts[1] + '}'
        sl.append(ts)
    print("~".join(sl) + r'\\')

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
    mae_person_sum = np.dot(which_person_arr, by - pre)
    mae_person_mean = mae_person_sum / which_person_arr.sum(axis=1)
    result = pd.DataFrame(mae_person_mean, columns=['残差（全体）'])
    return result


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
            ('1/x', FeatureTransformer(lambda x: 1 / x)),
            # ('sin', FeatureTransformer(np.sin)),
            # ('sin', FeatureTransformer(np.cos)),
            # ('norm', Normalizer()),
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression()),
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
    # degrees = [5, 7, 12, 18, 20]
    degrees = [2, 3, 4, 5]

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

    # kmeans
    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans_data = data[[col for col in data.columns if col.startswith('timestamp') or col.startswith('ED_volume')]].copy(deep=True)
    kmeans_data.fillna(0, inplace=True)
    kmeans.fit(kmeans_data.to_numpy())

    # 获取每个数据点的所属簇标签
    labels = kmeans.labels_
    print(labels)
    print(labels.shape)
    min_idx = labels
    # exit()

    # 创建t-SNE模型，将数据降维到2维
    tsne = TSNE(n_components=2, random_state=0)

    # 执行t-SNE降维
    X_tsne = tsne.fit_transform(kmeans_data)

    # 可视化聚类结果
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    # plt.colorbar(scatter, label='Class')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # # plt.title('t-SNE Visualization of Iris Dataset')
    # plt.show()

    # kmeans
    kmeans = KMeans(n_clusters=4, n_init=10)

    kmeans.fit(X_tsne)

    # 获取聚类中心的坐标
    cluster_centers = kmeans.cluster_centers_

    # 获取每个数据点的所属簇标签
    labels = kmeans.labels_
    print(labels)
    print(labels.shape)
    min_idx = labels

    # 可视化聚类结果
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, marker='X', c='red')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # # plt.title('K-Means Clustering')
    # plt.show()


    # exit(0)



    best_run = (0, mae_mean.min(axis=1).mean())
    step = 0
    while True:
        print("Step: {}".format(step))

        last_mae_curve = mae_curve.copy()

        # 用于记录每组有多少点
        curve_point_num = np.zeros(shape=(len(degrees)))

        if step == 17 or step == 5:
            font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
            plt.rcParams['font.family'] = font.get_name()
            fig, ax_arr = plt.subplots(ncols=len(degrees))

        for model_num, de in enumerate(degrees):
            which_data = which_person_arr[min_idx == model_num].sum(axis=0).astype(bool)
            # 如果该曲线没有分到点
            if not which_data.sum():
                continue
            # print(which_data)
            pre, model = fit_trans_poly(x[which_data], y[which_data], de)
            mae = mean_absolute_error(y[which_data], pre)

            if step == 17 or step == 5:

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

            print(de, mae, which_data.sum())
            mae_curve[model_num] = mae
            curve_point_num[model_num] = which_data.sum()

            maes[:, model_num] = np.abs(y - model.predict(x.reshape(-1, 1)))            # maes[:, model_num] = np.abs(np.exp(y) - np.exp(model.predict(x.reshape(-1, 1))))
            # print(pre)
        mae_mean = cal_which_group(which_person_arr, maes)
        min_idx = np.argmin(mae_mean, axis=1)

        if step == 17 or step == 5:
            plt.show()
            print(mae_mean.shape)
            print(min_idx)

            z = np.zeros(min_idx.shape[0]).astype(mae_mean.dtype)
            for i in range(min_idx.shape[0]):
                z[i] = mae_mean[i, min_idx[i]]
            print(z)
            print(z.shape)
            result = pd.DataFrame(np.stack([min_idx + 1, z], axis=1), columns=["所属亚组", "残差（亚组）"])
            return result


        if best_run[1] > mae_mean.min(axis=1).mean():
            best_run = (step, mae_mean.min(axis=1).mean())
        print(best_run)
        print((step, mae_mean.min(axis=1).mean()))
        print()

        if (mae_curve == last_mae_curve).all():
            bad_curve = np.argmax(mae_curve)
            bad_idx_bool = min_idx == bad_curve
            # print(bad_idx_bool)

            # min_idx[bad_idx_bool] = list(range(len(degrees))) * (bad_idx_bool.sum() // len(degrees)) + list(range(bad_idx_bool.sum() % len(degrees)))

            if curve_point_num.min() == curve_point_num[bad_curve]:
                # degrees[bad_curve] = min(degrees) - 1
                pass
            elif curve_point_num.max() == curve_point_num[bad_curve]:
                # degrees[bad_curve] = max(degrees) + 1
                pass
            else:
                # min_idx[bad_idx_bool] = list(range(len(degrees))) * (bad_idx_bool.sum() // len(degrees)) + list(range(bad_idx_bool.sum() % len(degrees)))
                move_list = list(range(len(degrees)))
                move_list.remove(bad_curve)
                min_idx[bad_idx_bool] = (move_list * (bad_idx_bool.sum() // len(move_list)) +
                                         move_list[:bad_idx_bool.sum() % len(move_list)])

        # time.sleep(0.2)
        step += 1
        # break


def q2_c():
    which = "ED"
    # which = "HM"
    data = q2_c_data(which=which)

    font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
    plt.rcParams['font.family'] = font.get_name()

    class_data = pd.read_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q2b.csv')

    data['所属亚组'] = class_data['所属亚组']
    data['残差（亚组）'] = class_data['残差（亚组）']

    print(data)
    print(data.shape)

    Y_NAME = f'{which}_volume_delta'

    data.fillna(np.nan, inplace=True)

    # fig, ax_arr = plt.subplots(ncols=4)
    for idx, group in list(data.groupby('所属亚组')) + [('all', data)]:
        print(idx, group.shape)

        flatten_data = pd.DataFrame(columns=['timestamp', f'{which}_volume', 'ID'], dtype=float)
        for j, _ in group.iterrows():
            for i in range(1, 9):
                a = group.loc[j, f'timestamp.{i}']
                if not pd.isna(a):
                    flatten_data = flatten_data._append({
                        'ID': group.loc[j, 'ID'],
                        'timestamp': group.loc[j, f'timestamp.{i}'],
                        f'{which}_volume': group.loc[j, f'{Y_NAME}.{i}'],
                        "脑室引流": group.loc[j, '脑室引流'],
                        "止血治疗": group.loc[j, '止血治疗'],
                        "降颅压治疗": group.loc[j, '降颅压治疗'],
                        "降压治疗": group.loc[j, '降压治疗'],
                        "镇静、镇痛治疗": group.loc[j, '镇静、镇痛治疗'],
                        "止吐护胃": group.loc[j, '止吐护胃'],
                        "营养神经": group.loc[j, '营养神经'],
                    }, ignore_index=True)

        fig, ax_arr = plt.subplots(ncols=4, nrows=2)
        for cure_idx, cure_name in enumerate(["脑室引流", "止血治疗", "降颅压治疗", "降压治疗", "镇静、镇痛治疗", "止吐护胃", "营养神经"]):
            pipeline = [
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('lr', LinearRegression())
            ]
            pipeline = Pipeline(pipeline)

            x = flatten_data[['timestamp', cure_name]].copy(deep=True)
            y = flatten_data[f'{which}_volume'].copy(deep=True)

            # print(y[np.isnan(y)])
            # exit(0)
            x.loc[:, 'timestamp'] = x['timestamp'].apply(np.log)
            y[y > 0] = np.log(y + 1)
            y[y < 0] = -np.log(-y - 1)
            sorted_idx = x['timestamp'].to_numpy().argsort()

            pipeline.fit(x, y)
            pre = pipeline.predict(x)

            ax = ax_arr[cure_idx // 4][cure_idx % 4]

            # plt.plot(x['timestamp'].to_numpy()[sorted_idx], pre[sorted_idx])

            pltx = x['timestamp']
            pltx = np.linspace(start=pltx.min(), stop=pltx.max(), num=1000)
            plty0 = pipeline.predict(np.stack([pltx, np.zeros(pltx.shape)], axis=1))
            plty1 = pipeline.predict(np.stack([pltx, np.ones(pltx.shape)], axis=1))
            # pltx = np.exp(pltx)

            # np.exp(x['timestamp'].to_numpy())
            ax.scatter(x['timestamp'], y, c=x[cure_name], s=3)
            ax.plot(pltx, plty0, c='red', label="{}=0".format(cure_name))
            ax.plot(pltx, plty1, c='green', label="{}=1".format(cure_name))
            ax.plot(pltx, np.zeros(pltx.shape), c='black')
            ax.set_xlabel('log(发病时间)')
            ax.set_ylabel('f(水肿体积变化值)')
            ax.legend()

        ax = ax_arr[1][3]
        ax.scatter([0], [0], c='purple', label='没接收该治疗的患者')
        ax.scatter([0], [0], c='yellow', label='接收了该治疗的患者')
        ax.scatter([0], [0], c='white')
        ax.scatter([1], [1], c='white')
        ax.scatter([-1], [-1], c='white')
        ax.annotate('其中$f= \\{ $', xy=(-0.53, 0))
        ax.annotate('$log(x+1), x \geq 0;$', xy=(-0.1, 0.06))
        ax.annotate('$-log(-x-1), others.$', xy=(-0.1, -0.06))
        ax.axis('off')
        ax.legend(loc='lower right')
        plt.subplots_adjust(left=0.06, right=0.98, top=0.956, bottom=0.076)
        plt.show()
        # exit()




def q2_d_q3_b():
    data = q2_d_data()
    flatten_data = pd.DataFrame(dtype=float)

    endswith_0 = [col for col in data.columns if col.endswith('.0')]
    endswith_0_x = [col for col in data.columns if col.endswith('.0_x')]
    endswith_0_y = [col for col in data.columns if col.endswith('.0_y')]

    for j, _ in data.iterrows():
        for i in range(1, 9):
            name_dict = {}
            name_dict.update({
                col: f"{col[:-1]}{i}" for col in endswith_0
            })
            name_dict.update({
                col: f"{col[:-3]}{i}{col[-2:]}" for col in endswith_0_x
            })
            name_dict.update({
                col: f"{col[:-3]}{i}{col[-2:]}" for col in endswith_0_y
            })
            cur_dict = {
                'ID': data.loc[j, 'ID'],
                # 'timestamp': data.loc[j, f'timestamp.{i}'],
                # 'ED_volume': data.loc[j, f'ED_volume.{i}'],
                # 'HM_volume': data.loc[j, f'HM_volume.{i}'],
                'HM_volume_delta': data.loc[j, f'HM_volume.{i}'] - data.loc[j, f'HM_volume.0'],
                'ED_volume_delta': data.loc[j, f'ED_volume.{i}'] - data.loc[j, f'ED_volume.0'],
                "脑室引流": data.loc[j, '脑室引流'],
                "止血治疗": data.loc[j, '止血治疗'],
                "降颅压治疗": data.loc[j, '降颅压治疗'],
                "降压治疗": data.loc[j, '降压治疗'],
                "镇静、镇痛治疗": data.loc[j, '镇静、镇痛治疗'],
                "止吐护胃": data.loc[j, '止吐护胃'],
                "营养神经": data.loc[j, '营养神经'],
                "90天mRS": data.loc[j, '90天mRS'],
            }
            cur_dict.update({
                key: data.loc[j, name_dict[key]] for key in name_dict
            })

            flatten_data = flatten_data._append(cur_dict, ignore_index=True)

    # font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
    # plt.rcParams['font.family'] = font.get_name()
    # fig, ax_arr = plt.subplots(ncols=2)
    # for i, method in enumerate(['spearman', 'pearson']):
    #     aaa = flatten_data[["HM_volume.0", "ED_volume.0", "HM_volume_delta", "ED_volume_delta"]].copy(deep=True)
    #     aaa.rename({
    #         "HM_volume.0": "血肿体积",
    #         "ED_volume.0": "水肿体积",
    #         "HM_volume_delta": "血肿体积变化值",
    #         "ED_volume_delta": "水肿体积变化值",
    #     }, axis=1, inplace=True)
    #     aaa = aaa.corr(method=method)
    #     sns.heatmap(aaa, ax=ax_arr[i])
    #     ax_arr[i].set_title(method + '相关系数')
    # plt.show()
    # exit()

    Y_COLUMN_NAME = '90天mRS'
    # Y_COLUMN_NAME = 'HM_volume.0'
    # Y_COLUMN_NAME = 'ED_volume.0'
    group_data = []
    for id_, g in flatten_data.groupby('ID'):
        g.drop('ID', axis=1, inplace=True)
        g.fillna(-123.0, inplace=True)
        group_data.append(g.to_numpy().astype(np.float32))

    Y_COLUMN_IDX = np.where(g.columns == Y_COLUMN_NAME)[0][0]
    group_data = np.stack(group_data, axis=0)
    print(group_data.shape)


    # corr = flatten_data[['timestamp', 'ED_volume', 'HM_volume']].corr()
    # sns.heatmap(corr)
    # plt.show()
    # corr = flatten_data[['timestamp', 'ED_volume', 'HM_volume']].corr(method="kendall")
    # sns.heatmap(corr)
    # plt.show()
    # corr = flatten_data[['timestamp', 'ED_volume', 'HM_volume']].corr(method="spearman")
    # sns.heatmap(corr)
    # plt.show()

    # plt.scatter(flatten_data['ED_volume'], flatten_data['HM_volume'])
    # plt.show()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Masking
    import tensorflow as tf
    import random
    seed_value = 42
    # 1. 设置 numpy 随机种子
    np.random.seed(seed_value)
    # 2. 设置 random 随机种子
    random.seed(seed_value)
    # 3. 设置 TensorFlow 随机种子
    tf.random.set_seed(seed_value)

    # 1. 准备数据
    # 假设您有一个包含单个维度的时序数据的DataFrame df。

    # 2. 数据预处理
    # 将数据划分为特征（X）和目标（y）

    X = group_data[:, :, list(range(Y_COLUMN_IDX)) + list(range(Y_COLUMN_IDX + 1, group_data.shape[-1]))]  # 去掉最后一个时间点，作为特征
    y = group_data[:, :, Y_COLUMN_IDX]  # 下一个时间点的值，作为目标

    # 数据归一化（可选）

    scalers = [MinMaxScaler() for _ in range(group_data.shape[2])]
    for i, s in enumerate(scalers[:-1]):
        X[:, :, i] = s.fit_transform(X[:, :, i])

    y[y == -123.] = 3.
    y = scalers[-1].fit_transform(y)

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = X[:100], X[130:], y[:100], y[130:]
    # X_train, X_test, y_train, y_test = X[:80], X[80:100], y[:80], y[80:100]

    # 3. 构建LSTM模型
    model = Sequential()
    model.add(Masking(mask_value=-123.0, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(1024, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(8))  # 输出层

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')  # 使用均方误差作为损失函数

    # 4. 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # 5. 评估模型
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    print("训练集损失:", train_loss)
    print("测试集损失:", test_loss)

    # 6. 进行预测
    y_pred = model.predict(X_test)

    # 如果进行了数据归一化，可以反归一化预测结果

    y_pred = scalers[-1].inverse_transform(y_pred)
    y_test = scalers[-1].inverse_transform(y_test)
    print(mean_absolute_error(y_pred, y_test))
    print(mean_squared_error(y_pred, y_test))
    label = abs(y_pred[:, -1].reshape(-1, 1) - np.arange(0, 7)).argmin(axis=1)
    print(label)


    yp = model.predict(X_train)
    yp = scalers[-1].inverse_transform(yp)

    lp = abs(yp[:, -1].reshape(-1, 1) - np.arange(0, 7)).argmin(axis=1)
    print(lp)
    a = np.concatenate([lp, [np.nan] * 30, label], axis=0)
    result = pd.DataFrame(a, columns=["预测mRS"])
    return result



if __name__ == "__main__":
    # q2_a().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q2a.csv')
    # q2_b().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q2b.csv')
    # q2_c()
    q2_d_q3_b().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q3b.csv')

