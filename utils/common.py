#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        :
# * Email         :
# * Description   : 公用的一些接口 Some common interfaces
# * Last modified :
# * *******************************************************
# * Filename      : common.py
import tensorflow as tf
import numpy as np


def printbar():
    """
    模型训练过程中的打印函数
    Print function in the training process of model
    """
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join(
        [timeformat(hour), timeformat(minite), timeformat(second)], separator=":"
    )
    return timestring


def cosine_similiarity(vec_left, vec_right):
    """
    余弦相似度:
    一个向量空间中两个向量夹角间的余弦值作为衡量两个个体之间差异的大小，
    余弦值接近1，夹角趋于0，表明两个向量越相似，余弦值接近于0，夹角趋于90度，
    表明两个向量越不相似。
    
    Cosine similarity:
    The cosine value of the included angle between two vectors in 
    a vector space is used to measure the difference between two individuals. 
    The cosine value is close to 1 and the included angle tends to 0, 
    indicating that the more similar the two vectors are, 
    the cosine value is close to 0 and the included angle tends to 90 degrees, 
    indicating that the less similar the two vectors are.

    :param vec_left:
    :param vec_right:
    :return:
    """
    vec_left = np.asarray(vec_left)
    vec_right = np.asarray(vec_right)
    num = np.dot(vec_left, vec_right)
    denom = np.linalg.norm(vec_left) * np.linalg.norm(vec_right)
    cos = -1 if denom == 0 else num / denom
    return cos


def euclidean_distance(vec_left, vec_right):
    """
    欧式距离
    最易于理解的一种距离计算方法，源自欧氏空间中两点间的距离公式

    Euclidean distance:
    One of the most comprehensible distance calculation methods is 
    derived from the distance formula between two points in Euclidean space.

    :param vec_left:
    :param vec_right:
    :return:
    """
    vec_left = np.asarray(vec_left)
    vec_right = np.asarray(vec_right)

    sq = np.square(vec_left - vec_right)
    dist = np.sqrt(np.sum(sq))
    return dist


def corr_coef(vec_left, vec_right):
    """
    皮尔逊距离距离
    相关系数的取值范围是[-1,1]。相关系数的绝对值越大，则表明X与Y相关度越高。当X与Y线性相关时，相关系数取值为1（正线性相关）或-1（负线性相关）

    Pearson distance:
    The range of correlation coefficient is [-1,1]. 
    The larger the absolute value of correlation coefficient, 
    the higher the correlation between X and Y. 
    When X is linearly related to Y, the correlation coefficient is 
    1 (positive linear correlation) or -1 (negative linear correlation).

    :param vec_left:
    :param vec_right:
    :return:
    """
    vec_left = np.asarray(vec_left)
    vec_right = np.asarray(vec_right)
    x_ = vec_left - np.mean(vec_left)
    y_ = vec_right - np.mean(vec_right)
    dist = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))

    return dist


if __name__ == "__main__":
    a, b = [1, 1, 0, 0], [1, 0, 0, 0]

    print(cosine_similiarity(a, b))
    print(euclidean_distance(a, b))
    print(corr_coef(a, b))
