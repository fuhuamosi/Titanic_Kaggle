# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from data_preprocess.view_data import get_train_set, get_test_set
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

__author__ = 'fuhuamosi'


# 使用随机森林填补Age属性
def set_missing_ages(df: DataFrame, age_rfr: RandomForestRegressor = None):
    numerical_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
    age_df = df[numerical_features]
    known_age = age_df[pd.notnull(age_df.Age)].as_matrix()
    unknown_age = age_df[pd.isnull(age_df.Age)].as_matrix()
    x = known_age[:, :-1]
    y = known_age[:, -1]
    if age_rfr is None:
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(x, y)
    else:
        rfr = age_rfr
    predict_ages = rfr.predict(unknown_age[:, :-1])
    df.loc[pd.isnull(df.Age), ['Age']] = predict_ages
    return rfr, df


# 使用随机森林填补Fare属性
def set_missing_fares(df: DataFrame):
    numerical_features = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
    age_df = df[numerical_features]
    known_fare = age_df[pd.notnull(age_df.Fare)].as_matrix()
    unknown_fare = age_df[pd.isnull(age_df.Fare)].as_matrix()
    x = known_fare[:, :-1]
    y = known_fare[:, -1]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    predict_fares = rfr.predict(unknown_fare[:, :-1])
    df.loc[pd.isnull(df.Fare), ['Fare']] = predict_fares
    return df


# 填补Cabin属性
def set_cabin_type(df: DataFrame):
    df.loc[(df.Cabin.notnull()), ['Cabin']] = 'Yes'
    df.loc[(df.Cabin.isnull()), ['Cabin']] = 'No'
    return df


# 设置虚拟变量(独热编码)
def set_dummy_vars(df: DataFrame):
    df.drop(labels=['Name', 'Ticket'], axis=1, inplace=True)
    discrete_features = list(df.dtypes[df.dtypes == 'object'].index)
    discrete_features.append('Pclass')
    dummies = [pd.get_dummies(df[f], prefix=f) for f in discrete_features]
    dummies.insert(0, df)
    df = pd.concat(dummies, axis=1)
    df.drop(labels=discrete_features, axis=1, inplace=True)
    return df


def scale_features(df: DataFrame):
    spec_features = ['Age', 'Fare']
    scaler = StandardScaler()
    for sf in spec_features:
        scale_param = scaler.fit(df[sf].reshape(-1, 1))
        df[sf + '_scaled'] = scaler.fit_transform(df[sf].reshape(-1, 1), scale_param)
    df.drop(labels=spec_features, axis=1, inplace=True)
    return df


def clean_train_set():
    train_set = get_train_set()
    age_rfr, train_set = set_missing_ages(train_set)
    train_set = set_cabin_type(train_set)
    train_set = set_dummy_vars(train_set)
    train_set = scale_features(train_set)
    return age_rfr, train_set


def clean_test_set():
    age_rfr, train_set = clean_train_set()
    test_set = get_test_set()
    _, test_set = set_missing_ages(test_set, age_rfr)
    test_set = set_missing_fares(test_set)
    test_set = set_cabin_type(test_set)
    test_set = set_dummy_vars(test_set)
    test_set = scale_features(test_set)
    train_set.to_csv('../dataset/cleaned_train.csv', index=False)
    test_set.to_csv('../dataset/cleaned_test.csv', index=False)
    return train_set, test_set


if __name__ == '__main__':
    clean_test_set()
