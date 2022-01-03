# -*- coding: utf-8 -*-
import os
import sys

o_path = os.getcwd()  # 返回当前工作目录 Return to the current working directory
sys.path.append(o_path)  # 添加自己指定的搜索路径 Add your own specified search path.
sys.path.append("../")

import pickle
import tensorflow as tf
import utils.common as common


def calcu_similar_user(user_id, top_k, user_features, dist_type):
    """
    相似用户
    Similar users

    :param user_id:
    :param top_k:
    :param user_features:
    :param dist_type:
    :return:
    """
    similiarities = {}
    user_feature = user_features[user_id]

    for ind, val in enumerate(user_features):
        if ind == user_id:
            continue
        if dist_type == "cosine_similiarity":
            similiarities[ind] = common.cosine_similiarity(user_feature, val)
            reverse_flag = True
        elif dist_type == "cosine_similiarity":
            similiarities[ind] = common.cosine_similiarity(user_feature, val)
            reverse_flag = False
        elif dist_type == "corr_coef":
            similiarities[ind] = common.corr_coef(user_feature, val)
            reverse_flag = False
        else:
            similiarities[ind] = common.cosine_similiarity(user_feature, val)
            reverse_flag = False

    sorted_res = sorted(
        similiarities.items(), key=lambda item: item[1], reverse=reverse_flag
    )
    return sorted_res[:top_k]


def calcu_similar_movie(movie_id, top_k, movie_features, dist_type):
    """
    相似电影
    Similar movies
    
    :param movie_id:
    :param top_k:
    :param movie_features:
    :param dist_type:
    :return:
    """
    similiarities = {}
    item_feature = movie_features[movie_id]

    for ind, val in enumerate(movie_features):
        if ind == movie_id:
            continue
        if dist_type == "cosine_similiarity":
            similiarities[ind] = common.cosine_similiarity(item_feature, val)
            reverse_flag = True
        elif dist_type == "cosine_similiarity":
            similiarities[ind] = common.cosine_similiarity(item_feature, val)
            reverse_flag = False
        elif dist_type == "corr_coef":
            similiarities[ind] = common.corr_coef(item_feature, val)
            reverse_flag = False
        else:
            similiarities[ind] = common.cosine_similiarity(item_feature, val)
            reverse_flag = False

    sorted_res = sorted(
        similiarities.items(), key=lambda item: item[1], reverse=reverse_flag
    )
    return sorted_res[:top_k]


if __name__ == "__main__":
    dist_type = "cosine_similiarity"

    # 读取user向量embedding 
    # loading the user Embedding vector
    with open("./data_resource/pickl_data/user_embdeeing.data", "rb") as uf:
        user_features = pickle.load(uf, encoding="utf-8")
    # 计算相似用户 
    # Calculating the similarity of users
    similar_users = calcu_similar_user(5900, 5, user_features, dist_type)
    print(similar_users)

    # 读取所有用户 
    # loading all users 
    with open("./data_resource/pickl_data/all_user.data", "rb") as uf:
        all_user = pickle.load(uf, encoding="utf-8")
    for id, dis in similar_users:
        print(id, all_user[str(id)])

    # 读取item向量embedding 
    # loading the item Embedding vector
    with open("./data_resource/pickl_data/item_embdeeing.data", "rb") as uf:
        item_features = pickle.load(uf, encoding="utf-8")
    # 计算相似电影 
    # Calculating the similarity of items
    similar_item = calcu_similar_movie(2, 5, item_features, dist_type)

    print(similar_item)
    # 读取所有item 
    # loading all items
    with open("./data_resource/pickl_data/all_item.data", "rb") as uf:
        all_item = pickle.load(uf, encoding="utf-8")
    for id, dis in similar_item:
        print(id, all_item[str(id)])
