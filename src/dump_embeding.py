#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import pickle
import tensorflow as tf
import utils.config as config


ckpt_path = config.checkpoint_path


def get_variable(value_name):
    """
    获取所有需要保存的变量
    Get all variables that need to be saved
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_path)  # 保存ckpt文件的文件夹 Folder where ckpt files are saved
    if ckpt and ckpt.model_checkpoint_path:
        reader = tf.pywrap_tensorflow.NewCheckpointReader(
            "data_resource/checkpoint/model.ckpt-1"
        )
        value = reader.get_tensor(value_name)
        print(value.shape)
        print(value)
        print(type(value))
        print("*" * 100)
        return value
    else:
        print("No checkpoint file found")


def dump_data(data, path):
    with open(path, "wb") as meta:
        pickle.dump(data, meta)


if __name__ == "__main__":
    user_embed_name = "user_id_embed/id_embed_matrix"  # 用户embedding的变量名字 The variable name of the user embedding
    data = get_variable(user_embed_name)
    user_embdeeing = "./data_resource/pickl_data/user_embdeeing.data"
    dump_data(data, user_embdeeing)

    item_embed_name = "item_embed/item_embed_matrix"  # item embedding的变量名字 The variable name of the item embedding
    data = get_variable(item_embed_name)
    item_embdeeing = "./data_resource/pickl_data/item_embdeeing.data"
    dump_data(data, item_embdeeing)
