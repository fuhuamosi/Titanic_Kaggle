# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import linear_model, cross_validation
import pandas as pd
import numpy as np
from pandas import DataFrame

__author__ = 'fuhuamosi'


def modeling(train_set: DataFrame):
    train_df = train_set.filter(regex='Survived|SibSp|Parch|Sex_*|Cabin_*|'
                                      'Embarked_*|Pclass_*|Age_scaled|Fare_scaled')
    train_np = train_df.as_matrix()
    x = train_np[:, 1:]
    y = train_np[:, 0]
    model = linear_model.LogisticRegression(penalty='l1')
    model.fit(x, y)
    print('The accuracy on train set is',
          cross_validation.cross_val_score(model, x, y, cv=5))
    return model, train_df


def predict(test_set: DataFrame, model: linear_model.LogisticRegression):
    test_df = test_set.filter(regex='SibSp|Parch|Sex_*|Cabin_*|'
                                    'Embarked_*|Pclass_*|Age_scaled|Fare_scaled')
    test_np = test_df.as_matrix()
    predictions = model.predict(test_np)
    result = DataFrame({'PassengerId': test_set['PassengerId'].as_matrix(),
                        'Survived': predictions.astype(np.int32)})
    result.to_csv('../predictions/logistic_regression.csv', index=False)


def view_coef(model, train_df):
    coef_list = list(model.coef_.T)
    abs_coef_list = [abs(c) for c in coef_list]
    coef_df = DataFrame({'columns': list(train_df.columns[1:]),
                         'coef': coef_list,
                         'abs_coef': abs_coef_list})
    coef_df.sort_values(by=['abs_coef'], ascending=[0], inplace=True)
    print(coef_df)


if __name__ == '__main__':
    lr_model, train = modeling(pd.read_csv('../dataset/cleaned_train.csv'))
    view_coef(lr_model, train)
    # predict(pd.read_csv('../dataset/cleaned_test.csv'), lr_model)
