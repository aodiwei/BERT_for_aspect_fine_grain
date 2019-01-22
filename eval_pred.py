#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '19-1-21'
# 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

pre_path = '/tmp/aspect_output/test_results.tsv'
lable_path = '/home/adw/workspace/study/dataset/aspect_bert/dev.tsv'

value_list = [-2, -1, 0, 1]
value_dict = {
    0: -2,
    1: -1,
    2: 0,
    3: 1,
}


def process(row):
    #     print(row)
    vals = row.values
    val_split = np.split(vals, 20)
    idxs = []
    for i, x in enumerate(val_split):
        idx = np.argmax(x)
        #         print(x)
        idxs.append('{}_{}'.format(i, value_dict[idx]))

    return idxs


def process_lab(row):
    label = row[0]
    label = label.split(',')
    return label


def eval():
    """

    :return:
    """
    df_pre = pd.read_csv(pre_path, sep='\t', header=None)
    df_label = pd.read_csv(lable_path, sep='\t', header=None)

    assert df_label.shape[0] == df_pre.shape[0]

    pred_rows = df_pre.apply(process, axis=1)
    pres = np.array(pred_rows.values.tolist())

    label_rows = df_label.apply(process_lab, axis=1)
    lebels = np.array(label_rows.values.tolist())

    f = []
    for i in range(20):
        fs = f1_score(lebels[:, i], pres[:, i], average='macro')

        f.append(fs)

    f = np.round(f, decimals=4)
    print('*********eval result*************')
    print(f)
    ret = np.mean(f)
    ret = np.round(ret, decimals=5)
    print(ret)

    return ret
