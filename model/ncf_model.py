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


def ncf_model_fn(features, labels, mode, params):
    """
    features: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    labels: train_input_fn的返回,注意train_input_fn函数返回值的顺序
    mode: tf.estimator.ModeKeys实例的一种
    params: 在初始化estimator时,传入的参数列表。dict形式,当前存入了特征的预定义 feature column,一些超参,一些特定的路径

    Features: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Labels: the return of train_input_fn. Pay attention to the order of return values of the train _ input _ fn function. 
    Mode: TF. Estimator. One of the modekeys instances 
    Params: the list of parameters passed in when initializing the estimator. 
    Dict form, currently stored in the predefined feature column of features, some super parameters and some specific paths.
    """
    lr = params["learning_rate"]
    dropout = params["dropout"]
    reg_gamma = params["reg_gamma"]
    is_reg = params["is_reg"]
    id_embed_size = params["dim_id"]

    # the switcher of activation function
    activation_func = "ReLU"
    if activation_func == "ReLU":
        activation_func = tf.nn.relu
    elif activation_func == "Leaky_ReLU":
        activation_func = tf.nn.leaky_relu
    elif activation_func == "ELU":
        activation_func = tf.nn.elu

    # the switcher of initializer
    initializer = "Normal"
    if initializer == "Normal":
        initializer = tf.truncated_normal_initializer(stddev=0.01)
    elif initializer == "Xavier_Normal":
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.glorot_uniform_initializer()

    regularizer = tf.contrib.layers.l2_regularizer(reg_gamma)
    # regularizer = tf.contrib.layers.l1_regularizer(reg_gamma)

    uid_slotid, uid_num = params["uid_nums"]
    item_slotid, item_num = params["itemid_nums"]

    uids_vocab = list(str(i) for i in range(uid_num))
    items_vocab = list(str(i) for i in range(item_num))

    # 具体输入的uid和item_id的值 passing the value of uid and item_id
    uid_value = features[uid_slotid]
    item_value = features[item_slotid]

    oov_num = 10
    table_uid = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(uids_vocab), num_oov_buckets=oov_num, default_value=-1
    )
    table_item = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(items_vocab), num_oov_buckets=oov_num, default_value=-1
    )

    uid_index = table_uid.lookup(uid_value)
    item_index = table_item.lookup(item_value)

    with tf.name_scope("input"):
        user_onehot = tf.one_hot(uid_index, uid_num + oov_num, name="user_onehot")
        item_onehot = tf.one_hot(item_index, item_num + oov_num, name="item_onehot")

    with tf.name_scope("embed"):
        user_embed_GMF = tf.layers.dense(
            inputs=user_onehot,
            units=id_embed_size,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="user_embed_GMF",
        )

        item_embed_GMF = tf.layers.dense(
            inputs=item_onehot,
            units=id_embed_size,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="item_embed_GMF",
        )

        user_embed_MLP = tf.layers.dense(
            inputs=user_onehot,
            units=id_embed_size,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="user_embed_MLP",
        )
        item_embed_MLP = tf.layers.dense(
            inputs=item_onehot,
            units=id_embed_size,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="item_embed_MLP",
        )

    # l2 正则项 L2 regularizer
    reg_w1 = tf.contrib.layers.l2_regularizer(reg_gamma)(user_embed_GMF)
    reg_w2 = tf.contrib.layers.l2_regularizer(reg_gamma)(item_embed_GMF)
    reg_w3 = tf.contrib.layers.l2_regularizer(reg_gamma)(user_embed_MLP)
    reg_w4 = tf.contrib.layers.l2_regularizer(reg_gamma)(item_embed_MLP)

    # Matrix Factorization
    with tf.name_scope("GMF"):
        GMF = tf.multiply(user_embed_GMF, item_embed_GMF, name="GMF")

    # MLP
    with tf.name_scope("MLP"):
        interaction = tf.concat(
            [user_embed_MLP, item_embed_MLP], axis=-1, name="interaction"
        )

        layer1_MLP = tf.layers.dense(
            inputs=interaction,
            # units=id_embed_size * 2,
            units = 512,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="layer1_MLP",
        )
        layer1_MLP = tf.layers.dropout(layer1_MLP, rate=dropout)

        layer2_MLP = tf.layers.dense(
            inputs=layer1_MLP,
            # units=id_embed_size,
            units = 256,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="layer2_MLP",
        )
        layer2_MLP = tf.layers.dropout(layer2_MLP, rate=dropout)

        layer3_MLP = tf.layers.dense(
            inputs=layer2_MLP,
            # units=id_embed_size // 2,
            units = 128,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="layer3_MLP",
        )
        layer3_MLP = tf.layers.dropout(layer3_MLP, rate=dropout)

        layer4_MLP = tf.layers.dense(
            inputs=layer3_MLP,
            # units=id_embed_size // 2,
            units = 64,
            activation=activation_func,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="layer4_MLP",
        )
        layer4_MLP = tf.layers.dropout(layer4_MLP, rate=dropout)

    # the concatenation of Matrix Factorization and MLP
    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer4_MLP], axis=-1, name="concatenation")

        logits = tf.layers.dense(
            inputs=concatenation,
            units=1,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            name="predict",
        )

    model_out = tf.nn.sigmoid(logits)
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
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        loss_avg, global_step=global_step
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss_avg, train_op=opt)