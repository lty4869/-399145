import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score, recall_score, roc_auc_score
from xgboost import XGBRegressor

from data.load_data import q1_a_data, q1_b_data


def _is_hm(first_vol, next_vol):
    """
    是否发生血肿扩张(单位10^-3ml)
    """
    if next_vol - first_vol >= 6000.:
        return True
    elif next_vol >= first_vol * 1.33:
        return True
    else:
        return False



def q1_a():
    result = pd.DataFrame(columns=['ID',
                                   '首次影像检查流水号',
                                   '是否发生血肿扩张',
                                   '血肿扩张时间'])

    data = q1_a_data()
    # for i in data.columns:
    #     print(i)
    # print(data)

    for row_num in range(data.shape[0]):
        row = data.iloc[row_num]
        addition_time = row['发病到首次影像检查时间间隔'] * 3600.
        start_time = row['入院首次检查时间点']
        for t in range(1, 14):
            end_time = row[f'随访{t}时间点']


            if not isinstance(end_time, float) or addition_time + (end_time - start_time) > 48 * 3600.:
                # print(row['ID'], 0, t, row[f'HM_volume.{t}'], row[f'HM_volume'])
                result = result._append({
                    'ID': row['ID'],
                    '首次影像检查流水号': row['入院首次影像检查流水号'],
                    '是否发生血肿扩张': 0,
                    '血肿扩张时间': '',
                }, ignore_index=True)
                break

            all_time = addition_time + (end_time - start_time)
            if _is_hm(row['HM_volume'], row[f'HM_volume.{t}']):
                # print(row['ID'], 1, t, row[f'HM_volume.{t}'], row[f'HM_volume'])
                result = result._append({
                    'ID': row['ID'],
                    '首次影像检查流水号': row['入院首次影像检查流水号'],
                    '是否发生血肿扩张': 1,
                    '血肿扩张时间': "{:.2f}".format(all_time / 3600.)
                }, ignore_index=True)
                break
            else:
                # print(row[f'HM_volume.{t}'], row[f'HM_volume'])
                pass

            # exit(0)
    return result


def _q1_b_y():
    result = pd.DataFrame(columns=['ID',
                                   '是否发生血肿扩张',
                                   '发病后时间差',
                                   '扩张比例',
                                   '扩张值',
                                   ])

    data = q1_a_data(first_100=False)
    # for i in data.columns:
    #     print(i)
    # print(data)

    for row_num in range(data.shape[0]):
        row = data.iloc[row_num]
        addition_time = row['发病到首次影像检查时间间隔'] * 3600.
        start_time = row['入院首次检查时间点']

        last_result = {
                'ID': row['ID'],
                '是否发生血肿扩张': 0,
                '血肿扩张时间': 0.,
                '扩张比例': 0.,
                '扩张值': 0.,
            }
        for t in range(1, 14):
            end_time = row[f'随访{t}时间点']

            if not isinstance(end_time, float) or addition_time + (end_time - start_time) > 48 * 3600.:
                result = result._append(last_result, ignore_index=True)
                break

            all_time = addition_time + (end_time - start_time)
            # print(row['ID'], 1, t, row[f'HM_volume.{t}'], row[f'HM_volume'])
            last_result = {
                'ID': row['ID'],
                '是否发生血肿扩张': 0,
                '血肿扩张时间': all_time / 3600.,
                '扩张比例': row[f'HM_volume.{t}'] / row[f'HM_volume'] - 1,
                '扩张值': row[f'HM_volume.{t}'] - row[f'HM_volume'],
            }
            if _is_hm(row['HM_volume'], row[f'HM_volume.{t}']):
                last_result['是否发生血肿扩张'] = 1
                result = result._append(last_result, ignore_index=True)
                break

            # exit(0)
    return result


def q1_b():
    data = q1_b_data()
    data.fillna(0, inplace=True)
    # le = LabelEncoder()
    # for col in data.columns:
    #     if col == 'Unnamed: 0':
    #         continue
    #     if isinstance(data.loc[0, col], str):
    #         data[col] = le.fit_transform(data[col])


    for i in data.columns:
        print(i)

    # data = data[[
    #     'Unnamed: 0',
    #     '镇静、镇痛治疗',
    #     '房颤史',
    #     '脑室引流',
    #     '冠心病史',
    #     '卒中病史',
    #     '止血治疗',
    #     # '降颅压治疗',
    #     # '降压治疗',
    #     # '营养神经',
    #     # '舒张压',
    #     # 'ED_Cerebellum_L_Ratio',
    #     'NCCT_original_firstorder_Skewness',
    #     'HM_Cerebellum_L_Ratio',
    #     'ED_MCA_R_Ratio',
    #     'original_shape_Flatness',
    #     'original_shape_MajorAxisLength',
    #     'original_shape_Maximum2DDiameterColumn',
    #     'original_shape_Maximum2DDiameterSlice',
    #     # 'NCCT_original_firstorder_Kurtosis'
    # ]]

    tmp = _q1_b_y()

    # print(data.shape, tmp.shape)
    a = pd.merge(data, tmp, left_on='Unnamed: 0', right_on='ID')
    # print(a.shape)
    # exit(0)

    # a = a[:100]
    # a_die = a[a['是否发生血肿扩张'] == 1]
    # a_live = a[a['是否发生血肿扩张'] == 0]
    #
    # alllll = ['镇静、镇痛治疗',
    #     '房颤史',
    #     '脑室引流',
    #     '冠心病史',
    #     '卒中病史',
    #     '止血治疗',]
    # d = a_die.shape[0]
    # li = a_live.shape[0]
    # for i in alllll:
    #     print(i, a_die[i].to_numpy().sum(), d - a_die[i].to_numpy().sum())
    #     print(i, a_live[i].to_numpy().sum(), li - a_live[i].to_numpy().sum())
    #
    # print(a_die.shape)
    # print(a_live.shape)
    # exit(0)


    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts/STFANGSO.TTF")
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['font.size'] = 10
    xx = [
        # '镇静、镇痛治疗',
        'NCCT_original_firstorder_Skewness',
        # 'HM_Cerebellum_L_Ratio',
        'ED_MCA_R_Ratio',
        'original_shape_Flatness',
        'original_shape_MajorAxisLength',
        'original_shape_Maximum2DDiameterColumn',
        'original_shape_Maximum2DDiameterSlice',]
    ncol_ = 3
    degrees = [3, 3, 3, 2, 2, 3, 4, 4, 4, 2, 2, 4]
    for y_name, thre in zip(['扩张比例', '扩张值'], [0.33, 6000]):
        # y_name = '扩张比例'
        c1 = a[y_name]
        fig, ax_arr = plt.subplots(2, ncol_, figsize=(10, 5))
        for i, name in enumerate(xx):
            ax = ax_arr[i // ncol_][i % ncol_]
            ax.set_xlabel(name)
            ax.set_ylabel(y_name)

            x = a[name][:100]
            y = c1[:100]
            idx = np.where(y > thre)[0]

            x = x[idx].to_numpy().reshape(-1, 1)
            aa = a[name].to_numpy().reshape(-1, 1)

            pe = PolynomialFeatures(degree=degrees.pop(0))
            x = pe.fit_transform(x)
            aa = pe.transform(aa)

            lr = LinearRegression()

            lr.fit(x, y[idx])
            pre = lr.predict(aa)
            idx = np.argsort(a[name].to_numpy())

            # print(lr.coef_)
            # formula = "y={:.3e}x^2+{:.3e}x+{:.3e}".format(lr.coef_[2], lr.coef_[1], lr.intercept_)
            # print(lr.coef_)
            # print(lr.intercept_)
            print(name, y_name)
            bb = lr.coef_.copy()
            bb[0] = lr.intercept_
            print(bb)
            print("y={}".format('+'.join(['{:.3e}x^{}'.format(j, i) for i, j in enumerate(bb)])))
            # exit(0)

            ax.scatter(a[name][:100], c1[:100], c=['pink' if i > thre else 'blue' for i in c1][:100])
            # ax.scatter(a[name][100:], c1[100:], c=['red' if i > thre else 'green' for i in c1][100:])
            ax.plot(a[name].to_numpy()[idx], pre[idx], c='black')

            xmin, ymin = a[name].min(), c1.min()
            xmax, ymax = a[name].max(), c1.max()
            xj, yj = xmax - xmin, ymax - ymin
            ax.set_xlim(xmin - 0.1 * xj, xmax + 0.1 * xj)
            ax.set_ylim(ymin - 0.1 * yj, ymax + 0.1 * yj)

            data[name + y_name] = pre

        # plt.show()

    for i, name in enumerate(xx):
        data[name + 'prob'] = 1/ (1 + np.exp(-(
            1 / (1 / (data[name + '扩张比例'].to_numpy() / 0.33) +
                1 / (data[name + '扩张值'].to_numpy() / 6000.))
        )))

    # exit(0)

    from answer.feature_add import add_NCCT_original_firstorder_Skewness

    # a = add_NCCT_original_firstorder_Skewness(a)
    # data['Skewness_engineer'] = a

    y = q1_a()['是否发生血肿扩张'].astype(int)
    print(y)

    # plt.scatter(data['NCCT_original_firstorder_Kurtosis'],
    #          data['NCCT_original_firstorder_Skewness'],
    #          c=y)
    # plt.show()
    # exit()

    train_set = data[data['Unnamed: 0'] < 'sub101'].drop('Unnamed: 0', axis=1)
    test_set = data.drop('Unnamed: 0', axis=1)
    print(train_set, test_set)

    train_set = pd.get_dummies(train_set)
    test_set = pd.get_dummies(test_set)

    # pf = PolynomialFeatures(degree=2)
    # test_set = pf.fit_transform(test_set)
    # print(test_set.shape)
    # exit(0)
    # ohe = OneHotEncoder()
    # test_set = ohe.fit_transform(test_set)
    # train_set = ohe.transform(train_set)

    # from draw.grid_test import grid_regression
    # a = grid_regression(test_set[:100], y[:100], None)
    #
    # print(a)
    # exit(0)

    # model_name = DecisionTreeRegressor
    # {'criterion': 'squared_error', 'max_depth': 3, 'max_leaf_nodes': None, 'min_samples_leaf': 3, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0, 'random_state': 42, 'splitter': 'random'}
    # {'criterion': 'squared_error', 'max_depth': 4, 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0, 'splitter': 'best', 'random_state': 42}
    # model_name = GradientBoostingRegressor
    # {'criterion': 'friedman_mse', 'learning_rate': 0.001, 'loss': 'squared_error', 'max_depth': 9, 'max_features': 'sqrt', 'n_estimators': 210, 'random_state': 42}

    model_name = RandomForestRegressor
    # {'criterion': 'friedman_mse', 'max_depth': 15, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 2, 'random_state': 42}

    # grid_search = GridSearchCV(
    #     model_name(),
    #     # {
    #     #     'criterion': ["squared_error", "absolute_error"],
    #     #     'splitter': ["best", "random"],
    #     #     # 'max_features': [None, "log2", "sqrt", "auto"],
    #     #     # 'max_depth': [None, 10, 30, 50, 70, 90, 100],
    #     #     'max_depth': range(3, 30),
    #     #     'min_samples_split': [2, 3, 5, 10],
    #     #     'min_samples_leaf': [1, 3, 5],
    #     #     'min_weight_fraction_leaf': [0],
    #     #     'max_leaf_nodes': [None],
    #     #     'random_state': [42],
    #     # },
    #     {
    #         'n_estimators': list(range(2, 3)),
    #         'max_depth': [i * 2 + 1 for i in range(1, 6)],
    #         'learning_rate': [10 ** -i for i in range(0, 4)],
    #         # 'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    #         # 'criterion': ['friedman_mse', 'squared_error'],
    #         # 'max_features': ['auto', 'sqrt', 'log2'],
    #         'random_state': [233]
    #     },
    #     verbose=1,
    #     n_jobs=5,
    #     cv=3,
    #     refit=True,
    # ).fit(test_set[:100], y[:100])
    # last_para = grid_search.best_params_
    # print(last_para)
    # dtr = model_name(**last_para)

    # dtr = model_name(**{'criterion': 'squared_error', 'max_depth': 3, 'max_leaf_nodes': None, 'min_samples_leaf': 3, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0, 'random_state': 42, 'splitter': 'random'})
    # dtr = model_name(**{'criterion': 'squared_error', 'learning_rate': 0.001, 'loss': 'squared_error', 'max_depth': None, 'max_features': 'log2', 'n_estimators': 10, 'random_state': 42})
    dtr = model_name(**{'criterion': 'friedman_mse', 'max_depth': 15, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 2, 'random_state': 42})
    dtr.fit(train_set, y[:100])
    dtr.fit(test_set[:100], y[:100])
    pre = dtr.predict(test_set)


    print(y[:100].tolist(), '\n', pre[:100])
    # print(y[100:130].tolist(), '\n', pre[100:130])
    print(y[130:].tolist(), '\n', pre[130:])

    mse = mean_squared_error(y[:100], pre[:100])
    r2 = roc_auc_score(y[:100], pre[:100])
    print(mse, r2)
    mse = mean_squared_error(y[100:130], pre[100:130])
    # r2 = roc_auc_score(y[100:130], pre[100:130])
    print(mse)
    mse = mean_squared_error(y[130:], pre[130:])
    r2 = roc_auc_score(y[130:], pre[130:])
    print(mse, r2)
    # f1 = f1_score(y[130:], pre[130:], average='micro')
    # rec = recall_score(y[130:], pre[130:])
    # print(f1, rec)

    result = pd.DataFrame(columns=['血肿扩张预测概率'])

    result['血肿扩张预测概率'] = pre

    return result



if __name__ == "__main__":
    # q1_a()
    # q1_a().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q1_a.csv')
    q1_b().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q1_b.csv')

