#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        : Henry Ren
# * Email         :
# * Description   : DNN模型代码 Code for DNN model
# * Last modified :
# * *******************************************************
# * Filename      : dnn_model.py
from __future__ import print_function

import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import tensorflow as tf

import utils.common as common
import utils.sub_module as sub_module


def dnn_model_fn(features, labels, mode, params):
    """
    dnn 模型具体模型实现
    features: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    labels: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    mode: tf.estimator.ModeKeys实例的一种
    params: 在初始化estimator时,传入的参数列表。dict形式,当前存入了特征的预定义 feature column,一些超参,一些特定的路径
    Implementation of dnn model  
    Features: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Labels: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Mode: TF. Estimator. One of the modekeys instances 
    Params: the list of parameters passed in when initializing the estimator. 
    Dict form, currently stored in the predefined feature column of features, some super parameters and some specific paths.
    """

    lr = params["learning_rate"]

    uid_slotid, _ = params["uid_nums"]
    item_slotid, _ = params["itemid_nums"]

    is_reg = params["is_reg"]
    reg_gamma = params["reg_gamma"]
    layer_nodes = params["layer_nodes"]
    all_feat_column = params["all_feat_col"]

    all_features = {
        k: v
        for k, v in features.items()
        if k.isdigit() and k not in (uid_slotid, item_slotid)
    }

    # 设置batch_norm在预测的时候不初始化 Set batch_norm not to initialize when predicting.

    if mode == tf.estimator.ModeKeys.TRAIN:
        bn_flag = True
    else:
        bn_flag = False

    user_embed_layer, item_embed_layer = sub_module.id_emb_init(
        features, params
    )  # id的embedding向量 embedding vectors of id


    reg_w1 = tf.contrib.layers.l2_regularizer(reg_gamma)(user_embed_layer)
    reg_w2 = tf.contrib.layers.l2_regularizer(reg_gamma)(item_embed_layer)  

    """开始建立模型 Start modeling"""

    with tf.name_scope("input_layer"):
        input = tf.feature_column.input_layer(
            all_features, all_feat_column, trainable=True
        )
        input = tf.concat([input, user_embed_layer], axis=1)
        input = tf.concat([input, item_embed_layer], axis=1)

    len_layers = len(layer_nodes)  # 共多少层 layers number
    with tf.variable_scope("dnn_deep_layer"):
        dnn_dense = tf.layers.dense(
            inputs=input, units=layer_nodes[0], activation=tf.nn.relu
        )
        for i in range(1, len_layers):
            dnn_dense = tf.layers.dense(
                inputs=dnn_dense, units=layer_nodes[i], activation=tf.nn.relu
            )
            # dnn_dense = tf.layers.batch_normalization(dnn_dense, training=bn_flag)
            dnn_dense = tf.contrib.layers.layer_norm(dnn_dense)
        model_out = tf.layers.dense(inputs=dnn_dense, units=1)
    model_out = tf.nn.sigmoid(model_out)
    model_out = tf.reduce_sum(model_out, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"label": features.get("label"), "scores": model_out}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions
            )
        }
        prediction_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs,
        )
        return prediction_spec

    rating = labels["label"]
    one = tf.ones_like(rating)  # 生成与 rating 大小一致的值全部为1的矩阵 Generate a matrix with all 1 values consistent with the rating size.
    zero = tf.zeros_like(rating)  # 生成与 rating 大小一致的值全部为0的矩阵 Generate a matrix with all 0 values  consistent with the rating size.
    label_threshold = params["label_threshold"]
    label = tf.where(rating > label_threshold, x=one, y=zero)

    if params["loss"] == "log_loss":
        loss = tf.losses.log_loss(labels=label, predictions=model_out)
    if not is_reg:
        loss_avg = tf.reduce_mean(loss, name="loss_avg")
    else:
        loss_avg = tf.reduce_mean(loss, name="loss_avg") 

    # loss_avg = tf.reduce_mean(loss, name="loss_avg")

    auc_value, auc_update_op = tf.metrics.auc(
        labels=label, predictions=model_out, name="auc"
    )

    predicted_classes = tf.to_int32(model_out > 0.5)
    precision, precision_update_op = tf.metrics.precision(
        labels=label, predictions=predicted_classes, name="precision"
    )
    recall, recall_update_op = tf.metrics.recall(
        labels=label, predictions=predicted_classes, name="recall"
    )

    f1_score, f1_update_op = tf.metrics.mean(
        (2 * precision_update_op * recall_update_op)
        / (precision_update_op + recall_update_op),
        name="f1_score",
    )
    label_mean = tf.reduce_mean(label)
    predict_mean = tf.reduce_mean(model_out)

    global_step = tf.train.get_global_step()
    model_type = params["model_type"]
    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            model_type + "_precision_eval": (precision, precision_update_op),
            model_type + "_recall_eval": (recall, recall_update_op),
            model_type + "_f1_eval": (f1_score, f1_update_op),
            model_type + "_auc_eval": (auc_value, auc_update_op),
        }

        loss_avg = tf.Print(
            loss_avg,
            [
                common.printbar(),
                global_step,
                "loss_avg",
                loss_avg,
                "label_mean:",
                label_mean,
                "predict_mean:",
                predict_mean,
                "auc",
                auc_update_op,
                "f1_score",
                f1_update_op,
                "precision",
            ],
            message=model_type + "_test:",
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=loss_avg, eval_metric_ops=eval_metric_ops
        )
    else:
        tf.summary.scalar(model_type + "_auc_train", auc_update_op)
        tf.summary.scalar(model_type + "_precision_train", precision_update_op)
        tf.summary.scalar(model_type + "_recall_train", recall_update_op)
        tf.summary.scalar(model_type + "_f1_train", f1_update_op)
        loss_avg = tf.Print(
            loss_avg,
            [
                common.printbar(),
                global_step,
                "loss_avg",
                loss_avg,
                "label_mean:",
                label_mean,
                "predict_mean:",
                predict_mean,
                "auc",
                auc_update_op,
                "f1_score",
                f1_update_op,
                "precision",
            ],
            message=model_type + "_train:",
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        loss_avg, global_step=global_step
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss_avg, train_op=opt)
