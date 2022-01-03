#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import tensorflow as tf

import utils.common as common


def mf_model_fn(features, labels, mode, params):
    """
    features: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    labels: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    mode: tf.estimator.ModeKeys实例的一种
    params: 在初始化estimator时,传入的参数列表。dict形式,当前存入了特征的预定义 feature column,一些超参,一些特定的路径

    Features: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Labels: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Mode: TF. Estimator. One of the modekeys instances Params: the list of parameters passed in when initializing the estimator. 
    Dict form, currently stored in the predefined feature column of features, some super parameters and some specific paths.
    """
    lr = params["learning_rate"]
    reg_gamma = params["reg_gamma"]
    is_reg = params["is_reg"]
    dim_id = params["dim_id"]

    uid_slotid, uid_num = params["uid_nums"]
    item_slotid, item_num = params["itemid_nums"]

    uids_vocab = list(str(i) for i in range(uid_num))
    items_vocab = list(str(i) for i in range(item_num))

    # 具体输入的uid和item_id的值 Passing the uid and item_id values
    uid_value = features[uid_slotid]
    item_value = features[item_slotid]

    oov_num = 10

    # 用户和item的embeddings矩阵 Embeddings matrix of users and items
    uid_embs = tf.Variable(
        tf.truncated_normal([uid_num + oov_num, dim_id]),
        # tf.contrib.layers.xavier_initializer([uid_num + oov_num, dim_id]),
        dtype=tf.float32,
        trainable=True,
        name="uid_emb_matrix",
    )
    item_embs = tf.Variable(
        tf.truncated_normal([item_num + oov_num, dim_id]),
        # tf.contrib.layers.xavier_initializer([item_num + oov_num, dim_id]),        
        dtype=tf.float32,
        trainable=True,
        name="item_emb_matrix",
    )

    # 用户和item的得分偏置 the user's rating bias of item.
    uids_bias_mat = tf.Variable(
        tf.truncated_normal([uid_num + oov_num, 1]),
        # tf.contrib.layers.xavier_initializer([uid_num + oov_num, 1]),
        dtype=tf.float32,
        trainable=True,
        name="uid_bias_matrix",
    )
    items_bias_mat = tf.Variable(
        tf.truncated_normal([item_num + oov_num, 1]),
        # tf.contrib.layers.xavier_initializer([item_num + oov_num, 1]),
        dtype=tf.float32,
        trainable=True,
        name="item_bias_matrix",
    )

    table_uid = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(uids_vocab), num_oov_buckets=oov_num, default_value=-1
    )
    table_item = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(items_vocab), num_oov_buckets=oov_num, default_value=-1
    )

    is_hash = False
    if is_hash:
        # 用户和item的id的hash,结果是一个int64或者一个tensor。和参数形状保持一致
        # the hash of the id of the user and item, and the result is an int64 or a tensor. Consistent with the parameter shape.
        uid_index = tf.string_to_hash_bucket_fast(uid_value, uid_num)
        item_index = tf.string_to_hash_bucket_fast(item_value, item_num)
    else:
        uid_index = table_uid.lookup(uid_value)
        item_index = table_item.lookup(item_value)

    # 用户和item的embedding,embedding矩阵通过index值查找到 
    # the embedding of user and item, and can be found by index value.
    uid_emb = tf.nn.embedding_lookup(uid_embs, uid_index)
    item_emb = tf.nn.embedding_lookup(item_embs, item_index)

    # 用户和item的偏置项,偏置矩阵通过index值查找到 
    # the bias of user and item, and the bias matrix can be found through the index value.
    uid_bias = tf.nn.embedding_lookup(uids_bias_mat, uid_index)
    item_bias = tf.nn.embedding_lookup(items_bias_mat, item_index)

    # l2 正则项 L2 regularizer
    reg_w1 = tf.contrib.layers.l2_regularizer(reg_gamma)(uid_emb)
    reg_w2 = tf.contrib.layers.l2_regularizer(reg_gamma)(item_emb)

    # Matrix multiplication, inner prodiction of user_id and item_id
    model_out = tf.multiply(uid_emb, item_emb)
    model_out = tf.reduce_sum(model_out, axis=1)
    model_out = tf.nn.sigmoid(model_out)

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
    zero = tf.zeros_like(rating)  # 生成与 rating 大小一致的值全部为0的矩阵 Generate a matrix with all 0 values consistent with the rating size.
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
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
        loss_avg, global_step=global_step
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss_avg, train_op=opt)
