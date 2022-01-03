#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 预定于数据类型代码  Feature data types code
# * Last modified :
# * *******************************************************
# * Filename      : feature_column.py
from __future__ import print_function

import os
import sys

sys.path.append("../")
sys.path.append(os.getcwd())  # 添加自己指定的搜索路径 Add your own specified search path.

import tensorflow as tf
from collections import defaultdict
import numpy as np


# UserID
# + "\t"
# + Gender
# + "\t"
# + Age
# + "\t"
# + Occupation
# + "\t"
# + MovieID
# + "\t"
# + Title
# + "\t"
# + Genres
# + "\t"
# + Rating


def make_column(confs):
    """
    制作相应的column
    Make the corresponding feature column.
    """
    all_slotid = confs["all_slotid"]
    slotid_name = confs["slotid_name"]
    uid_slotid, _ = confs["uid_nums"]
    item_slotid, _ = confs["itemid_nums"]
    # 特征列 feature column
    all_column_list = []
    for slotid in all_slotid:
        if slotid == uid_slotid or slotid == item_slotid:
            continue

        column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=slotid,
            vocabulary_file=confs["vocabulary_file_path"] + "/" + slotid_name[slotid],
            num_oov_buckets=5,
        )
        column = tf.feature_column.embedding_column(
            column, confs["dim_other"], combiner="mean"
        )
        all_column_list.append(column)
    return all_column_list


if __name__ == "__main__":

    confs = {
        "model_type": "dnn",
        "layer_nodes": [32, 16],
        "learning_rate": 0.001,
        "epoch_num": 1,
        "batch_size": 1024,
        "dim": 16,  # vectore size
        "l2_reg": 0.0001,
        "loss": "log_loss",
        "eval_sample_count": 2048,
        "train_path": "data_resource/sample.csv",
        "test_path": "data_resource/sample.csv",
        "model_path": "data_resource/savemodel",
        "checkpoint_path": "data_resource/checkpoint",
        "all_slotid": ["4", "5", "6", "7", "8"],
    }

    make_column(confs)
