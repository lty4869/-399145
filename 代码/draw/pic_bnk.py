import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import acf

from data.load_data import q1_b_data
from answer.q1 import q1_a, _q1_b_y

def to_timestamp_ignore_errors(x):
    try:
        return pd.to_datetime(x).timestamp()
    except Exception:
        return x

def q1_b_corr():

    def heat_map_2x1(df, input_columns, output_columns):
        # 设置使用的字体1
        font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
        plt.rcParams['font.family'] = font.get_name()
        plt.rcParams['font.size'] = 14

        import re
        input_columns = [s for s in input_columns if not re.search(r'[\u4e00-\u9fff]', s)]

        df = df[output_columns + input_columns]
        
        # df.head()
        df_corr = df.corr()
        # df_corr = df_corr.loc[output_columns][input_columns]
        df_corr = df_corr.loc[input_columns][output_columns]
        fig, ax_arr = plt.subplots(1, 4, figsize=(8, 6))

        df_corr_list = []
        sep = df_corr.shape[0] // 4 + 1
        for i in range(0, df_corr.shape[0], sep):
            # df_corr_cur = df_corr.iloc[:, i:i + sep]
            df_corr_cur = df_corr.iloc[i:i + sep, :]
            df_corr_list.append(df_corr_cur)

        # annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
        for i in range(len(df_corr_list)):
            # print([df_corr_list[i][j] for j in df_corr_list[i]])
            sns.heatmap(df_corr_list[i],
                        annot=False,
                        cbar=True if i == 3 else False,
                        vmin=min([min(pp.to_numpy()) for pp in df_corr_list]),
                        vmax=max([max(pp.to_numpy()) for pp in df_corr_list]),
                        square=True,
                        cmap="Blues",
                        fmt='.2g',
                        ax=ax_arr[i],
                        )
        plt.show()

    data = q1_b_data()
    data.fillna(0, inplace=True)
    # le = LabelEncoder()
    # for col in data.columns:
    #     if col == 'Unnamed: 0':
    #         continue
    #     if isinstance(data.loc[0, col], str):
    #         data[col] = le.fit_transform(data[col])

    data = data.drop('Unnamed: 0', axis=1)
    data = pd.get_dummies(data)
    for i in data.columns:
        print(i)

    y = q1_a()['是否发生血肿扩张']
    print(y)

    from data.load_data import translate_eng2chs
    data.rename(columns={i: translate_eng2chs(i) for i in data.columns}, inplace=True)
    data.rename(columns={i: i.replace('original_', '').replace('firstorder_', '') for i in data.columns}, inplace=True)
    print(type(data))
    print(data.columns)
    x = data.columns.tolist()
    # x.remove('Unnamed: 0')
    # x = x[:len(x) // 2]
    data['whether hematoma expansion occurred'] = y
    heat_map_2x1(data, x, ['whether hematoma expansion occurred'])

    a = _q1_b_y()
    data.drop('whether hematoma expansion occurred', axis=1, inplace=True)
    data['hematoma expansion caused by ratio'] = a['扩张比例']
    heat_map_2x1(data, x, ['hematoma expansion caused by ratio'])

    data.drop('hematoma expansion caused by ratio', axis=1, inplace=True)
    data['hematoma expansion caused by value'] = a['扩张值']
    heat_map_2x1(data, x, ['hematoma expansion caused by value'])

    exit(0)



from data.load_data import q2_ab_data


def q2_a_d_choose():
    font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['font.size'] = 20

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

    sort_idx = np.argsort(x)

    plt.scatter(x, y)
    plt.xlabel('发病时间(s)')
    plt.ylabel('水肿体积(10$^{-3}$ml)')
    plt.grid()
    plt.show()

    x = np.log(x)
    y = np.log(y)

    plt.scatter(x, y)
    plt.xlabel('log(发病时间)')
    plt.ylabel('log(水肿体积)')
    plt.grid()
    plt.show()

    degree_range = np.arange(2, 50)
    degree_mse = np.zeros(shape=degree_range.shape)
    for degree in degree_range:
        pe = PolynomialFeatures(degree=degree)
        xx = pe.fit_transform(x.reshape(-1, 1))

        lr = LinearRegression()
        lr.fit(xx, y)
        pre = lr.predict(xx)


        by = np.exp(y)
        pre = np.exp(pre)


        mse = mean_absolute_error(by, pre)
        print(degree, mse)
        degree_mse[degree - 2] = mse

    plt.plot(degree_range, degree_mse, linewidth=2, alpha=0.5)
    plt.scatter(degree_range, degree_mse, s=50)

    for i, d in enumerate([3, 7, 18, 20, ]):
        plt.annotate("D={}, MAE={:.2f}".format(d, degree_mse[d - 2]),
                     xy=(d, float(degree_mse[d - 2])),
                     xytext=(d + i * 2, float(degree_mse[d - 2]) + 100 * (4 - i)),
                     arrowprops=dict(arrowstyle='->', color='orange', linewidth=2),
                     fontsize=20,
                     color='blue',)

    plt.grid()
    plt.xlabel('多项式函数维度D')
    plt.ylabel('平均绝对误差MAE')
    # plt.legend()

    plt.show()

    plt.rcParams['font.size'] = 15
    fig, ax_arr = plt.subplots(nrows=2, ncols=4)
    for i, j in enumerate([3, 7, 18, 20]):
        pe = PolynomialFeatures(degree=j)
        xx = pe.fit_transform(x.reshape(-1, 1))

        lr = LinearRegression()
        lr.fit(xx, y)
        pre = lr.predict(xx)

        ax = ax_arr[0][i]
        ax.scatter(x, y)
        ax.plot(x[sort_idx], pre[sort_idx], c='orange', label=f'D={j}')
        ax.set_xlabel('log(发病时间)')
        ax.set_ylabel('log(水肿体积)')
        ax.grid()
        ax.legend()

        bx = np.exp(x)
        by = np.exp(y)
        pre = np.exp(pre)

        ax = ax_arr[1][i]
        ax.scatter(bx, by)
        ax.plot(bx[sort_idx], pre[sort_idx], c='orange', label=f'D={j}')
        ax.set_xlabel('发病时间(s)')
        ax.set_ylabel('水肿体积(10$^{-3}$ml)')

        ax.grid()

        # mse = mean_absolute_error(by, pre)
        # print(mse)
        ax.legend()
    plt.show()



if __name__ == "__main__":

    q1_b_corr()
    # q2_a_d_choose()