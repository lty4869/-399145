import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['font.sans-serif'] = ['simsun']


def heat_map_2x2(df, input_columns, output_columns):
    df = df[output_columns + input_columns]
    # df.head()
    df_corr = df.corr().abs()
    df_corr = df_corr.loc[output_columns][input_columns]
    fig, ax_arr = plt.subplots(2, 2, figsize=(10, 5))

    df_corr_list = []
    sep = df_corr.shape[1] // 4 + 1
    for i in range(0, df_corr.shape[1], sep):
        df_corr_cur = df_corr.iloc[:, i:i + sep]
        df_corr_list.append(df_corr_cur)

    # annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
    for i in range(len(df_corr_list)):
        # print([df_corr_list[i][j] for j in df_corr_list[i]])
        sns.heatmap(df_corr_list[i], cbar=False, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g', ax=ax_arr[i // 2][i % 2])

    plt.show()


def heat_map_2x1(df, input_columns, output_columns):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # 设置使用的字体1
    font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['font.size'] = 8
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
                    vmin=-0.2,
                    vmax=0.25,
                    square=True,
                    cmap="Blues",
                    fmt='.2g',
                    ax=ax_arr[i],
                    )
    plt.show()


def bar_plot_2xn(df, columns, bins):
    for idx, column in enumerate(columns):
        pd_cur = pd.DataFrame(pd.cut(df[column], bins=bins, right=False))
        # pd_cur.set_index(pd_cur.index.astype('string'), drop=True)
        pd_cur[column + '_temp'] = 0
        pd_cur = pd_cur.groupby(by=column).count()
        if len(columns) != 1:
            plt.subplot(2, len(columns) // 2 + len(columns) % 2, idx + 1)
        plt.bar(
            [str(i).replace(', ', '\n') for i in pd_cur.index],
            [pd_cur.loc[i][column + '_temp'] for i in pd_cur.index],
            width=0.5,
            label=column,
        )
        plt.ylabel('number')
        plt.legend()
    plt.show()


def residual_plot(df, x_col, y_col, title=None):
    sns.residplot(x=x_col, y=y_col, data=df, lowess=True, color="g")
    if title:
        plt.title(title)
    plt.show()


def range_plot(df, x_col, y_gt_col, y_pred_col, range_):
    plt.plot(df[x_col], df[y_gt_col], 'g-.', label=y_gt_col)
    plt.plot(df[x_col], df[y_pred_col], 'b-o', label=y_pred_col)
    plt.plot(df[x_col], df[y_gt_col] + range_, 'r:.', label='upper bound')
    plt.plot(df[x_col], df[y_gt_col] - range_, 'r:.', label='lower bound')
    plt.legend()
    plt.show()


def scatter(df, column_i, column_o):
    corr = df.corr().abs()
    corr = corr.loc[column_o][column_i]
    corr_idx = corr.argsort()

    plt.rcParams['axes.facecolor'] = 'k'
    plt.figure(figsize=(11, 5))
    # cmap = ListedColormap(["#ffd700", "#0057b8", "#ecfa4b", "#bc10cb"])
    cmap = 'plasma'
    for i in range(3):
        column_one = corr_idx[corr_idx == i].index[0]
        column_two = column_o
        print(column_one, column_two)

        ax = plt.subplot(1, 3, i + 1)

        df_count = df[[column_one, column_two]].groupby(column_two).count()
        df_count = df_count / df_count.sum() * 3000.0
        # print(df.shape)
        # print(df[column_two].shape)
        # print([i for i in df[column_two].tolist()])
        # print(df[column_two].tolist())
        ax.scatter(df[column_one], df[column_two], s=df_count.loc[df[column_two]].iloc[:, 0],
                   c=df[column_two], cmap=cmap)
        ax.set_xlabel(column_one)
        ax.set_ylabel(column_two)
        # ax.set_aspect('equal')
        # ax0 = ax

    plt.tight_layout(w_pad=1.0)
    plt.show()
    plt.rcParams['axes.facecolor'] = '#0057b8' # blue


if __name__ == "__main__":
    import pickle
    def bin_read(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    from configs.config_cm_iron import output_columns, input_columns
    # data = pd.read_excel(r'E:\GongCheng\PyCharmProjectsAll\annealing_workmanship_process_monitor\data\cm_iron'
    #                    # r'\LC27CM_C512_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_1_17_05.xlsx')
    #                    # r'\LC27CM_C512_HARD_TEST_TYPE_H_R_B.xlsx')
    #                    r'\MI27CM_C008_C608_C708_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_2_20_01.xlsx')
    data = bin_read(r'E:\GongCheng\PyCharmProjectsAll\cm_iron\data\cm_iron'
           # r'\LC27CM_C512_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_1_17_05_SAMPLE_POS_CODE_0.bin')
           # r'\LC27CM_C512_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_1_17_05_SAMPLE_POS_CODE_9.bin')
           # r'\LC27CM_C512_HARD_TEST_TYPE_H_R_B_SAMPLE_POS_CODE_0.bin')
           # r'\LC27CM_C512_HARD_TEST_TYPE_H_R_B_SAMPLE_POS_CODE_9.bin')
           r'\MI27CM_C008_C608_C708_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_2_20_01_SAMPLE_POS_CODE_0.bin')
           # r'\MI27CM_C008_C608_C708_TEST_DIRECT_CODE_EL_GAUGE_CODE_SAMPLE_SHAPE_CODE_2_20_01_SAMPLE_POS_CODE_9.bin')

    from models.remove_outlier import box_plot
    old_shape = data.shape

    all_data = data
    # full_data = bin_read(r'E:\GongCheng\PyCharmProjectsAll\cm_iron\data\cm_iron\raw\LC27CM_C512.bin')
    full_data = bin_read(r'E:\GongCheng\PyCharmProjectsAll\cm_iron\data\cm_iron\raw\MI27CM_C008_C608_C708.bin')
    print(all_data.shape)
    columns_recorded = all_data.columns
    all_data = pd.merge(all_data, full_data, how='inner', on=list(set(all_data.columns) & set(full_data.columns)))
    print(all_data.shape)
    all_data = all_data[all_data['ST_NO'] == 'DP0942D1']
    data = all_data[columns_recorded]

    data = box_plot(data, list(set(output_columns) & set(data.columns)))
    print("去除了离群值{}行.".format(old_shape[0] - data.shape[0]))
    input_columns = list(set(input_columns) & set(data.columns))
    output_columns = list(set(output_columns) & set(data.columns))

    from configs.config_cm_iron import eng2chs
    input_columns = [eng2chs[i] for i in input_columns]
    output_columns = [eng2chs[i] for i in output_columns]

    data.rename(columns=eng2chs, inplace=True)

    bar_plot_2xn(data, output_columns, 10)
    heat_map_2x2(data, input_columns, output_columns)

    corr = data.corr()

    for col in output_columns:
        show_cols = list(corr.loc[col].argsort().index)
        print(show_cols)
        for j in output_columns:
            show_cols.remove(j)
        show_cols = show_cols[-3:]
        print(show_cols)
        scatter(data, show_cols, col)
    # pass

