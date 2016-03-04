# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

__author__ = 'fuhuamosi'


def tree_modeling(train_set: DataFrame, reg):
    train_df = train_set.filter(regex=reg)
    train_np = train_df.as_matrix()
    x = train_np[:, 1:]
    y = train_np[:, 0]
    clf = LogisticRegression(penalty='l1')
    clf.fit(x, y)
    scores = np.array(cross_validation.cross_val_score(clf, x, y, cv=5))
    print('The accuracy on train set is', scores.mean())
    return clf, train_df


def predict(test_set: DataFrame, model: LogisticRegression, reg, filename):
    test_df = test_set.filter(regex=reg)
    test_np = test_df.as_matrix()
    predictions = model.predict(test_np)
    result = DataFrame({'PassengerId': test_set['PassengerId'].as_matrix(),
                        'Survived': predictions.astype(np.int32)})
    result.to_csv(filename, index=False)


def view_coef(model: LogisticRegression, train_df):
    coef_list = list(model.coef_.T)
    coef_df = DataFrame({'columns': list(train_df.columns[1:]),
                         'coef': coef_list})
    coef_df['abs_coef'] = abs(coef_df['coef'])
    coef_df.sort_values(by=['abs_coef'], ascending=[0], inplace=True)
    print(coef_df)


if __name__ == '__main__':
    feature_reg = 'Sex|Age|Fare|Family|Kid|' \
                  'Embarked_[C|Q|S]|Pclass_[1-3]|' \
                  'Ticket'
    lr_model, sample_df = tree_modeling(pd.read_csv('../dataset/cleaned_train.csv'),
                                        reg='Survived|' + feature_reg)
    view_coef(lr_model, sample_df)
