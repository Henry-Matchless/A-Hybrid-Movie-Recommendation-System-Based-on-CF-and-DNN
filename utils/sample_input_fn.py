#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 样本输入代码 sample input code
# * Last modified :
# * *******************************************************
# * Filename      : sample_input_fn.py
from __future__ import print_function

import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import tensorflow as tf

import utils.config as config


def parse_fn(line, confs):

    """
    对样本数据每一行进行处理
    每一行具体的数据处理的方式如下
    Each row of sample data is proces. The specific data processing methods of each line are as follows
    """

    # 样本例子 Sample
    all_slotid = confs["all_slotid"]

    all_column_num = confs["all_column_num"]

    record_defaults = [
        ["0.0"] for _ in range(all_column_num)
    ]  # all_column_num 是训练数据所有列的数量 All_column_num is the number of all columns of training data.
    feats = tf.decode_csv(line, record_defaults, field_delim="\t")
    label_index = confs["label_index"]
    label = feats[label_index]

    common_fid_list = []
    common_features_list = []

    for _slotid in all_slotid:
        key = int(_slotid)
        if _slotid in confs["multi_value_index"]:
            values = tf.string_split([feats[key]], delimiter="|", skip_empty=True)
            common_fid_list.append(_slotid)
            common_features_list.append(values)
        else:
            common_fid_list.append(_slotid)
            common_features_list.append(feats[key])

    samples = dict(zip(common_fid_list, common_features_list))
    rating = tf.strings.to_number(label, out_type=tf.float32)
    labels = {"label": rating}

    return samples, labels


def train_input_fn(confs, mode="train", batch_size=1024, epoch_num=1, if_shuffle=True):
    """
    样本数据输入函数
    Sample data input function
    """

    if mode == "train":
        tfrecord_files = confs["train_path"]
    elif mode == "test":
        tfrecord_files = confs["test_path"]

    dataset = tf.data.TextLineDataset(tfrecord_files)

    # 数据跳过第一行 
    # Skipping the first row of data

    # dataset = dataset.skip(1)
    if if_shuffle:
        # 数据打散 Data scattering
        dataset = dataset.shuffle(buffer_size=1000)
    # 数据重复 Data repeating
    dataset = dataset.repeat(epoch_num)
    # 对每一行数据进行处理
    # Each row of data is processed
    dataset = dataset.map(lambda line: parse_fn(line, confs), num_parallel_calls=20)
    # 数据取batch 
    # Data fetch batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    # 缓冲管道大小
    # Buffer pipe size
    dataset = dataset.prefetch(buffer_size=200)

    return dataset


if __name__ == "__main__":
    confs = {
        "layer_nodes": config.layer_nodes,
        "model_type": config.model_type,
        "learning_rate": config.learning_rate,
        "epoch_num": config.epoch_num,
        "batch_size": config.batch_size,
        "loss": config.loss,
        "eval_sample_count": config.eval_sample_count,
        "train_path": config.train_path,
        "test_path": config.test_path,
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
        "all_slotid": config.all_slotid,
    }

    train_input_fn(confs)
