#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 导出模型函数 export the model function
# * Last modified :
# * *******************************************************
# * Filename      : export_model.py

from __future__ import print_function
import tensorflow as tf


def serving_input_receiver_fn(confs):
    """
    定义导出模型的输入输出
    Define the input and output of the export model.
    """
    all_slotid = confs["all_slotid"]
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="input_feat"
    )
    receiver_tensors = {"inputs": serialized_tf_example}
    feature_spec = {}
    feature_spec["sample"] = tf.FixedLenFeature([], dtype=tf.string)
    input_feat_json = tf.parse_example(serialized_tf_example, feature_spec)
    raw_input_sample = input_feat_json["sample"]

    all_column_num = confs["all_column_num"]
    record_defaults = [[""] for _ in range(all_column_num)]
    feats = tf.decode_csv(raw_input_sample, record_defaults, field_delim="\t")
    common_fid_list = []
    common_features_list = []
    for _slotid in all_slotid:
        key = int(_slotid)
        if _slotid in confs["multi_value_index"]:
            values = tf.string_split(
                feats[key], delimiter="|", skip_empty=True
            )  # 注意这地方与sample的区别 这里不需要加中括号 Note the difference between this place and sample. There is no need to add brackets here.

            common_fid_list.append(_slotid)
            common_features_list.append(values)
        else:
            common_fid_list.append(_slotid)
            common_features_list.append(feats[key])

    label_index = confs["label_index"]
    label = feats[label_index]
    rating = tf.strings.to_number(label, out_type=tf.float32)

    samples = dict(zip(common_fid_list, common_features_list))
    samples["label"] = rating

    return tf.estimator.export.ServingInputReceiver(samples, receiver_tensors)
