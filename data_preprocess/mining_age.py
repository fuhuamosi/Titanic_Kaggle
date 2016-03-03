# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from data_preprocess.view_data import get_train_set, get_test_set
import re
import math
import numpy as np

__author__ = 'fuhuamosi'


def match_element(element):
    # pattern = re.compile(r'.*M[r|i](s){0,2}\..*')
    pattern = re.compile(r".*Mr(s)?\..*")
    if math.isnan(element[1]):
        return False
    return pattern.match(element[0])


def age_info():
    train_set = get_train_set()
    train_np = train_set[['Name', 'Age']].as_matrix()
    temp_list = list(filter(match_element, train_np))
    temp_np = np.array([x[1] for x in temp_list])
    print(temp_np.mean(), temp_np.std())


def match_name(name, regex):
    return re.compile(regex).match(name)


def test_age(df: DataFrame):
    sub = 0
    for index, row in df.iterrows():
        name = row['Name']
        age = row['Age']
        if not math.isnan(age):
            if age <= 8:
                res = 'kid'
            elif age <= 30:
                res = 'young'
            elif age <= 45:
                res = 'middle'
            else:
                res = 'old'
        else:
            if match_name(name, r".*Master\..*"):
                res = 'kid'
            elif match_name(name, r".*Miss\..*"):
                res = 'young'
            elif match_name(name, r".*Mr(s)?\..*"):
                res = 'middle'
            else:
                res = 'young'
        df.loc[sub, 'Age'] = res
        sub += 1
    return df


if __name__ == '__main__':
    age_info()
    # test_age(get_train_set())
