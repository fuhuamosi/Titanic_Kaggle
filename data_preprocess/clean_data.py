# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from data_preprocess.view_data import get_train_set, get_test_set
from sklearn.preprocessing import StandardScaler
import re

__author__ = 'fuhuamosi'


# # 使用随机森林填补Age属性
# def set_missing_ages(df: DataFrame, age_rfr: RandomForestRegressor = None):
#     numerical_features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
#     age_df = df[numerical_features]
#     known_age = age_df[pd.notnull(age_df.Age)].as_matrix()
#     unknown_age = age_df[pd.isnull(age_df.Age)].as_matrix()
#     x = known_age[:, :-1]
#     y = known_age[:, -1]
#     if age_rfr is None:
#         rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#         rfr.fit(x, y)
#     else:
#         rfr = age_rfr
#     predict_ages = rfr.predict(unknown_age[:, :-1])
#     df.loc[pd.isnull(df.Age), ['Age']] = predict_ages
#     return rfr, df


# 填补Cabin属性
def set_cabin(df: DataFrame):
    df.loc[(df.Cabin.notnull()), ['Cabin']] = 1
    df.loc[(df.Cabin.isnull()), ['Cabin']] = 0
    df.Cabin = df.Cabin.astype('int')
    return df


# 设置虚拟变量(独热编码)
def set_dummy_vars(df: DataFrame):
    df.drop(labels=['Name'], axis=1, inplace=True)
    discrete_features = list(df.dtypes[df.dtypes == 'object'].index)
    discrete_features.append('Pclass')
    dummies = [pd.get_dummies(df[f], prefix=f) for f in discrete_features]
    dummies.insert(0, df)
    df = pd.concat(dummies, axis=1)
    df.drop(labels=discrete_features, axis=1, inplace=True)
    return df


def scale_features(df: DataFrame):
    spec_features = ['Fare']
    scaler = StandardScaler()
    for sf in spec_features:
        scale_param = scaler.fit(df[sf].reshape(-1, 1))
        df[sf + '_scaled'] = scaler.fit_transform(df[sf].reshape(-1, 1), scale_param)
    df.drop(labels=spec_features, axis=1, inplace=True)
    return df


def match_name(name, regex):
    return re.compile(regex).match(name)


def fill_ages(df: DataFrame):
    sub = 0
    df_part = df[['Name', 'Age']]
    for index, row in df_part.iterrows():
        name = row['Name']
        if match_name(name, r".*Master\..*"):
            res = 4.5
        elif match_name(name, r".*Miss\..*"):
            res = 22.0
        elif match_name(name, r".*Mr\..*"):
            res = 32.5
        elif match_name(name, r".*Mrs\..*"):
            res = 36.0
        else:
            res = 30.0
        df.loc[sub, 'Age'] = res
        sub += 1
    return df


def fill_embark(df):
    df.loc[df.Embarked.isnull(), ['Embarked']] = 'S'
    return df


def set_sex(df: DataFrame):
    df.loc[df.Sex == 'female', ['Sex']] = 1
    df.loc[df.Sex == 'male', ['Sex']] = 0
    df.Sex = df.Sex.astype('int')
    return df


def set_family(df):
    df['Family'] = df['SibSp'] + df['Parch']
    df.loc[df['Family'] > 0, ['Family']] = 1
    df.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
    return df


def add_kid_field(df):
    df['Kid'] = 0
    df.loc[df['Age'] <= 16, ['Kid']] = 1
    return df


def set_ticket(df):
    for index, row in df.iterrows():
        df.loc[index, ['Ticket']] = row['Ticket'][0]
    df['Ticket'] = df['Ticket'].astype('int')
    return df


def clean_data_set(df: DataFrame, filename):
    df = fill_ages(df)
    df = fill_embark(df)
    df = set_ticket(df)
    df = set_cabin(df)
    df = set_sex(df)
    df = set_family(df)
    df = set_dummy_vars(df)
    df = add_kid_field(df)
    # df = scale_features(df)
    df.to_csv(filename, index=False)
    return df


if __name__ == '__main__':
    clean_data_set(get_train_set(), '../dataset/cleaned_train.csv')
    clean_data_set(get_test_set(), '../dataset/cleaned_test.csv')
