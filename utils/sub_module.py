#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 一些公用的库 Some public utils function for DNN
# * Last modified :
# * *******************************************************
# * Filename      : sub_module.py
import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import utils.config as config
import tensorflow as tf


def id_emb_init(features, params):
    """
    id 和 物品的 embedding
    the Embedding of id and items
    """

    # reg_gamma = 0.01
    uid_slotid, uid_num = params["uid_nums"]
    item_slotid, item_num = params["itemid_nums"]
    dim_id = params["dim_id"]
    user_id = features[uid_slotid]  # 具体传入的id值 The specific id value passed in
    item_id = features[item_slotid]  # 具体传入的item值 The specific item value passed in
    user_id = tf.string_to_number(user_id, out_type=tf.int32)
    item_id = tf.string_to_number(item_id, out_type=tf.int32)

    with tf.variable_scope("user_id_embed"):
        user_id_embed_matrix = tf.get_variable(
            name="id_embed_matrix",
            shape=[uid_num + 1, dim_id],
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            # initializer=tf.contrib.layers.xavier_initializer(),
        )
        user_embed_layer = tf.nn.embedding_lookup(
            user_id_embed_matrix, user_id, name="uid_lookup"
        )

    with tf.variable_scope("item_embed"):
        movie_id_embed_matrix = tf.get_variable(
            name="item_embed_matrix",
            shape=[item_num + 1, dim_id],
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            # initializer=tf.contrib.layers.xavier_initializer(),
        )
        item_embed_layer = tf.nn.embedding_lookup(
            movie_id_embed_matrix, item_id, name="item_lookup"
        )

    # reg_w1 = tf.contrib.layers.l2_regularizer(reg_gamma)(user_embed_layer)
    # reg_w2 = tf.contrib.layers.l2_regularizer(reg_gamma)(item_embed_layer)    

    # return user_embed_layer, item_embed_layer, reg_w1, reg_w2
    return user_embed_layer, item_embed_layer
