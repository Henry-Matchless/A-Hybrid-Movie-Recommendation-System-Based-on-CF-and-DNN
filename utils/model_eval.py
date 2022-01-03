#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 对测试数据进行评估 Evaluate the test data
# * Last modified :
# * *******************************************************
# * Filename      : model_eval.py
from __future__ import print_function

import os
import time
import tensorflow as tf
from collections import defaultdict
from sklearn import metrics
import numpy as np


def predict(raw_feature, confs):
    """
    模型预测函数
    Model prediction function
    """
    model_dir = confs["model_path"]
    files = [file_name for file_name in tf.gfile.ListDirectory(model_dir)]
    files = sorted(files, reverse=True)

    timeArray = time.localtime(int(files[0]))
    model_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    timestamp_saved_model_dir = os.path.join(model_dir, files[0])
    predictor = tf.contrib.predictor.from_saved_model(timestamp_saved_model_dir)
    out_dict = defaultdict(lambda: [])
    labels_count = len(raw_feature)
    feature_proto_list = []
    for i in range(labels_count):
        line = "{}".format(raw_feature[i].decode("utf-8", "ignore"))
        features = {}
        features["sample"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[line.encode("utf-8") if type(line) == str else line]
            )
        )
        feature_proto = tf.train.Example(
            features=tf.train.Features(feature=features)
        ).SerializeToString()
        feature_proto_list.append(feature_proto)
    temp = predictor({"inputs": feature_proto_list})  # 会调用导出模型的时候的数据接收函数 Call the data receiving function when exporting the model.

    out_dict["label"] = temp["label"]
    out_dict["scores"] = temp["scores"]
    return out_dict, model_time


def load_eval_samples(confs):
    """
    载入评估样本
    Load the evaluation sample
    """
    tfrecord_files = confs["test_path"]
    dataset = tf.data.TextLineDataset(tfrecord_files)
    dataset = dataset.batch(confs["eval_sample_count"])
    iterator = dataset.make_one_shot_iterator()
    raw_feature = iterator.get_next()
    return raw_feature


def model_eval(confs):
    """
    模型评估代码
    Model evaluation code
    """
    sess = tf.Session()
    raw_feature = load_eval_samples(confs)
    raw_feature = sess.run(raw_feature)
    out_dict, model_time = predict(raw_feature, confs)

    raw_labels = out_dict["label"]
    raw_scores = out_dict["scores"]

    label_threshold = confs.get("label_threshold")
    predict_threshold = confs.get("predict_threshold")

    labels = np.where(raw_labels > label_threshold, 1, 0)

    scores = np.where(raw_scores > predict_threshold, 1, 0)

    all_accuracy = metrics.accuracy_score(labels, scores)

    auc = metrics.roc_auc_score(labels, raw_scores)

    C2 = metrics.confusion_matrix(labels, scores)

    TN_num = C2[0][0]
    FP_num = C2[0][1]
    FN_num = C2[1][0]
    TP_num = C2[1][1]

    f1_s = metrics.f1_score(labels, scores)
    recall_s = metrics.recall_score(labels, scores)

    print("*" * 65 + " == model eval == " + "*" * 65)
    print("=== model time :", model_time)
    print("=== model type :", confs.get("model_type"))
    print("=== epoch num  :", confs.get("epoch_num"))
    print("=== batch size :", confs.get("batch_size"))
    print("=== score mean   :", np.mean(out_dict["scores"]))
    print("=== score median :", np.median(out_dict["scores"]))
    print("=== the num of all eval sample :", len(labels))
    print("=== the num of pos eval sample :", (TP_num + FN_num))
    print("=== the num of neg eval sample :", (TN_num + FP_num))
    print("=== the TN of all smples :", TN_num)
    print("=== the FP of all smples :", FP_num)
    print("=== the FN of all smples :", FN_num)
    print("=== the TP of all smples :", TP_num)
    print("=== the all samples accuracy  :", all_accuracy)
    print("=== the Recall of all smples  :", recall_s)
    print("=== the F1score of all smples :", f1_s)
    print("=== the auc of test sample    :", auc)
    print("*" * 65 + " == model eval == " + "*" * 65)


if __name__ == "__main__":
    pass
