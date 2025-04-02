import numpy as np

from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier

import functools
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
import catboost as cb
from catboost import CatBoostRegressor
CatBoostRegressor = functools.partial(CatBoostRegressor, verbose=False, iterations=10)
CatBoostRegressor.__name__ = 'CatBoostRegressor'

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV

from functools import partial
from sklearn.metrics import make_scorer


def score_fun(y_test_cur, pred, output_range):
    ideal_rate = np.zeros(pred.shape)
    ideal_rate[np.where(pred <= y_test_cur + output_range)] += 0.5
    ideal_rate[np.where(pred >= y_test_cur - output_range)] += 0.5
    # print("{}/{}".format(ideal_rate.astype(np.int64).sum(), pred.shape[0]))
    ideal_rate = ideal_rate.astype(np.int64).sum() / pred.shape[0]
    return ideal_rate


def grid_regression(x_train, y_train, output_range):
    model_dict = {
        # KMeans: [
        #     {
        #         'n_clusters': list(range(2, 11)),
        #     }
        # ],

        # SVR: [
        #     # {
        #     #     'kernel': ['linear', 'rbf', 'sigmoid', 'precomputed'],
        #     #     'gamma': ['scale', 'auto'],
        #     # },
        #     {
        #         'kernel': ['poly'],
        #         'degree': list(range(2, 10)),
        #         'gamma': ['scale', 'auto'],
        #     }
        # ],

        DecisionTreeRegressor: [
            # https://blog.csdn.net/u013344884/article/details/79276825
            {
                'criterion': ["squared_error", "absolute_error"],
            }, {
                'splitter': ["best", "random"],
            }, {
            #     'max_features': [None, "log2", "sqrt", "auto"],
            # }, {
                'max_depth': [None, 10, 30, 50, 70, 90, 100] + list(range(2, 10)),
            }, {
                'min_samples_split': [2, 3, 5, 10],
            }, {
                'min_samples_leaf': [1, 3, 5],
            }, {
                'min_weight_fraction_leaf': [0],
            }, {
                'max_leaf_nodes': [None],
            }
        ],

        RandomForestRegressor: [
            # https://blog.csdn.net/xiaohutong1991/article/details/108178143
            {
                'n_estimators': list(range(10, 425, 10)) + list(range(2, 10)),
            }, {
                'max_depth': [i + 1 for i in range(1, 40)] + [None],
            }, {
                'min_samples_leaf': list(range(1, 10)),
            }, {
                'min_samples_split': list(range(2, 10)),
            }, {
            #     'max_features': ['auto', 'sqrt', 'log2'],
            # }, {
                'criterion': ['friedman_mse', 'squared_error'],
            },
        ],

        GradientBoostingRegressor: [
            {
                'n_estimators': list(range(10, 425, 10)) + list(range(2, 10)),
            }, {
                'max_depth': [i * 4 + 1 for i in range(1, 40)] + [None],
            }, {
                'learning_rate': [10 ** -i for i in range(0, 4)],
            }, {
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            }, {
                'criterion': ['friedman_mse', 'squared_error'],
            # }, {
            #     'max_features': ['auto', 'sqrt', 'log2'],
            },
        ],

        # XGBRegressor: [
        #     # https://zhuanlan.zhihu.com/p/95304498
        #     # https://www.zhihu.com/question/34470160/answer/2050672813
        #     {
        #         'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
        #     # }, {
        #         'n_estimators': list(range(20, 201, 50)) + list(range(2, 10)),
        #     }, {
        #         'max_depth': list(range(3, 22, 2)),
        #     }, {
        #         'booster': ['gbtree', 'dart'],
        #     }, {
        #         'gamma': [0, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        #     }, {
        #         'min_child_weight': [1, 3, 5, 7],
        #     }, {
        #         'lambda': [10 ** -i for i in range(-4, 4)],
        #     # }, {
        #         'alpha': [10 ** -i for i in range(-4, 4)],
        #     }, {
        #         'objective': ['reg:squarederror'],
        #     }, {
        #         'colsample_bytree': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
        #     },{
        #         'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #     },
        # ],

        LGBMRegressor: [
            # https://www.pythonheidong.com/blog/article/468277/6e714898cdbbb5af9979/
            {
                'learning_rate': [10 ** -i for i in range(0, 4)],
            }, {
                'n_estimators': list(range(20, 1001, 20)) + list(range(2, 10)),
            }, {
                'num_leaves': list(range(3, 120, 2)),
            }, {
                'max_depth': list(range(3, 21)) + [-1],
            }, {
                # 'boosting_type': ['gbdt', 'rf', 'dart', 'goss'],
            },
        ],

        # CatBoostRegressor: [
        #     # http://www.javashuo.com/article/p-vxvueirq-dc.html
        #     {
        #         'depth': list(range(3, 17)),
        #     }, {
        #         'learning_rate': [0.03, 0.1, 0.15],
        #     }, {
        #         'l2_leaf_reg': [1, 4, 9],
        #     },
        # ],

    }

    best_record = (-10000000, None)

    for model in model_dict:
        # para_sum = 0
        # for item in model_dict[model]:
        #     cur_sum = 1
        #     for key in item:
        #         cur_sum *= len(item[key])
        #     para_sum += cur_sum
        # if para_sum * 3 > x_train.shape[0]:
        #     x_train = np.concatenate([x_train] * (para_sum * 3 // x_train.shape[0] + 2), axis=0)
        #     y_train = np.concatenate([y_train] * (para_sum * 3 // y_train.shape[0] + 2), axis=0)
        #     print(x_train.shape)

        print("网格搜索模型: {}".format(model.__name__))
        if model == DecisionTreeRegressor or model == RandomForestRegressor or model == GradientBoostingRegressor\
                or model == LGBMRegressor or model == XGBRegressor or model == CatBoostRegressor:
            last_para = dict()
            for each_para in model_dict[model]:
                last_para = {k: [last_para[k]] for k in last_para}
                last_para.update(each_para)
                grid_search = GridSearchCV(
                    model(),
                    [last_para],
                    verbose=0,
                    n_jobs=5,
                    cv=3,
                    refit=True,
                    scoring='roc_auc',
                )
                grid_search.fit(x_train, y_train)
                last_para = grid_search.best_params_
                print(last_para)
            print(grid_search.best_score_)
        else:
            grid_search = GridSearchCV(
                model(),
                model_dict[model],
                verbose=0,
                n_jobs=2,
                cv=3,
                refit=True,
                scoring='roc_auc',
            )
            grid_search.fit(x_train, y_train)
            print('最佳评价指标: {}, 该模型最佳参数: {}'.format(grid_search.best_score_, grid_search.best_params_))
        # print(grid_search.best_estimator_)
        # best_model = grid_search.best_estimator_
        # best_model.fit(x_train, y_train)
        # pred = gbr.predict(x_test_cur)
        if grid_search.best_score_ > best_record[0]:
            best_record = (grid_search.best_score_, grid_search.best_estimator_)

    print(best_record[1])
    return best_record[1], best_record[0]




