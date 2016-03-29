# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from pandas import DataFrame

__author__ = 'fuhuamosi'


def leave_out(x: np.ndarray, y: np.ndarray):
    size = x.shape[0]
    train_size = int(1.0 * size)
    train_x, valid_x = x[:train_size, :], x[train_size:, :]
    train_y, valid_y = y[:train_size], y[train_size:]
    return train_x, train_y, valid_x, valid_y


def boost_modeling(train_set: DataFrame, reg):
    train_df = train_set.filter(regex=reg)
    train_np = train_df.as_matrix()
    x = train_np[:, 1:]
    y = train_np[:, 0]
    train_x, train_y, valid_x, valid_y = leave_out(x, y)
    dtrain = xgb.DMatrix(data=train_x, label=train_y)
    dvalid = xgb.DMatrix(data=valid_x, label=valid_y)
    watchlist = [(dtrain, 'train')]
    param = {'max_depth': 6, 'eta': 0.05, 'silent': 1,
             'objective': 'binary:logistic', 'subsample': 0.9}
    bst = xgb.train(param, dtrain, num_boost_round=17, evals=watchlist)
    return bst


def predict(test_set, model, reg, filename):
    test_df = test_set.filter(regex=reg)
    test_np = test_df.as_matrix()
    dtest = xgb.DMatrix(data=test_np)
    predictions = model.predict(dtest)
    predictions = np.array([int(p > 0.5) for p in predictions])
    result = DataFrame({'PassengerId': test_set['PassengerId'].as_matrix(),
                        'Survived': predictions.astype(np.int32)})
    result.to_csv(filename, index=False)


if __name__ == '__main__':
    train_data = pd.read_csv('../dataset/cleaned_train.csv')
    test_data = pd.read_csv('../dataset/cleaned_test.csv')
    feature_reg = 'Sex|Age|Family|Kid|Fare|Family' \
                  '|Embarked_[S]|Pclass_[3]' \
                  '|Ticket_[1|3]|Cabin_[X|B]'
    bst_model = boost_modeling(train_data, 'Survived|' + feature_reg)
    predict(test_data,
            bst_model,
            feature_reg,
            '../predictions/xgboost.csv')
