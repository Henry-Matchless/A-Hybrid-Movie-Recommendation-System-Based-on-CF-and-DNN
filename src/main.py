#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        : Henry Ren
# * Email         :
# * Description   : Main function of models
# * Last modified :
# * *******************************************************
# * Filename      : main.py
from __future__ import print_function

import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import numpy as np
import tensorflow as tf
import utils.sample_input_fn as sample_input_fn
import utils.feature_column as feature_column
import model.dnn_model as dnn_model
import model.mf_model as mf_model
import model.ncf_model as ncf_model


import utils.model_eval as model_eval
import utils.export_model as export_model
import utils.config as config

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ["VAR_PARTITION_THRESHOLD"] = "100000"


def get_params():
    """
    Get all the parameters required by the model
    """
    confs = {
        "label_index": config.label_index,
        "layer_nodes": config.layer_nodes,
        "model_type": config.model_type,
        "learning_rate": config.learning_rate,
        "epoch_num": config.epoch_num,
        "batch_size": config.batch_size,
        "batch_size_eval": config.batch_size_eval,
        "dim_id": config.dim_id,
        "dim_other": config.dim_other,
        "loss": config.loss,
        "eval_sample_count": config.eval_sample_count,
        "train_path": config.train_path,
        "test_path": config.test_path,
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
        "vocabulary_file_path": config.vocabulary_file_path,
        "all_column_num": config.all_column_num,
        "all_slotid": config.all_slotid,
        "user_slotid": config.user_slotid,
        "movie_slotid": config.movie_slotid,
        "uid_nums": config.uid_nums,
        "itemid_nums": config.itemid_nums,
        "slotid_name": config.slotid_name,
        "multi_value_index": config.multi_value_index,
        "label_threshold": config.label_threshold,
        "predict_threshold": config.predict_threshold,
        "reg_gamma": config.reg_gamma,
        "is_reg": config.is_reg,
        "dropout": config.dropout,
    }

    all_feat_col = feature_column.make_column(confs)
    confs["all_feat_col"] = all_feat_col
    return confs


def main(confs):
    """
    Set configuration
    Set model
    """
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={"GPU": 0, "CPU": 1}),
        log_step_count_steps=10,
        save_summary_steps=confs["epoch_num"] * 5,
        save_checkpoints_steps=confs["epoch_num"] * 5,
        save_checkpoints_secs=None,
        keep_checkpoint_max=5,
    )

    model_type = confs["model_type"]
    if model_type == "dnn":
        model = tf.estimator.Estimator(
            config=config,
            model_fn=dnn_model.dnn_model_fn,
            params=confs,
            model_dir=confs["checkpoint_path"],
        )
    elif model_type == "mf":
        model = tf.estimator.Estimator(
            config=config,
            model_fn=mf_model.mf_model_fn,
            params=confs,
            model_dir=confs["checkpoint_path"],
        )
    elif model_type == "ncf":
        model = tf.estimator.Estimator(
            config=config,
            model_fn=ncf_model.ncf_model_fn,
            params=confs,
            model_dir=confs["checkpoint_path"],
        )
    else:
        model = tf.estimator.Estimator(
            config=config,
            model_fn=dnn_model.dnn_model_fn,
            params=confs,
            model_dir=confs["checkpoint_path"],
        )

    # 这里面input_fn使用lambda方式的目的是为了给train_input_fn函数传参数 
    # in there, input_fn uses lambda mode to pass parameters to the train_input_fn function.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: sample_input_fn.train_input_fn(
            confs=confs,
            mode="train",
            batch_size=confs["batch_size"],
            epoch_num=confs["epoch_num"],
            if_shuffle=True,
        ),
        max_steps=None,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: sample_input_fn.train_input_fn(
            confs=confs,
            mode="test",
            batch_size=confs["batch_size_eval"],
            epoch_num=confs["epoch_num"],
            if_shuffle=False,
        ),
        steps=None,
        start_delay_secs=10,
        throttle_secs=0,
        # name="eval"
    )

    tf.estimator.train_and_evaluate(
        estimator=model, train_spec=train_spec, eval_spec=eval_spec
    )

    model.export_saved_model(
        confs["model_path"], lambda: export_model.serving_input_receiver_fn(confs)
    )
    model_eval.model_eval(confs)


if __name__ == "__main__":
    confs = get_params()
    main(confs)
    # model_eval.model_eval(confs)
