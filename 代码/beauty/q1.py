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
    tmp = _q1_b_y()
    # print(data.shape, tmp.shape)
    a = pd.merge(data, tmp, left_on='Unnamed: 0', right_on='ID')
    # print(a.shape)
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



    y = q1_a()['是否发生血肿扩张'].astype(int)
    print(y)


    train_set = data[data['Unnamed: 0'] < 'sub101'].drop('Unnamed: 0', axis=1)
    test_set = data.drop('Unnamed: 0', axis=1)
    print(train_set, test_set)


    model_name = RandomForestRegressor
    dtr = model_name(**{'criterion': 'friedman_mse', 'max_depth': 15, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 2, 'random_state': 42})
    dtr.fit(train_set, y[:100])
    dtr.fit(test_set[:100], y[:100])
    pre = dtr.predict(test_set)


    print(y[:100].tolist(), '\n', pre[:100])
    print(y[130:].tolist(), '\n', pre[130:])

    mse = mean_squared_error(y[:100], pre[:100])
    r2 = roc_auc_score(y[:100], pre[:100])
    print(mse, r2)
    mse = mean_squared_error(y[100:130], pre[100:130])
    print(mse)
    mse = mean_squared_error(y[130:], pre[130:])
    r2 = roc_auc_score(y[130:], pre[130:])
    print(mse, r2)

    result = pd.DataFrame(columns=['血肿扩张预测概率'])

    result['血肿扩张预测概率'] = pre

    return result



if __name__ == "__main__":
    # q1_a().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q1_a.csv')
    q1_b().to_csv(r'E:\GongCheng\PyCharmProjectsAll\shuxuejianmo\results\q1_b.csv')

