#!/usr/bin/python
# -*- coding: utf-8 -*-
# /********************************************************
# * Author        : Henry Ren
# * Email         :
# * Description   : 数据清洗的代码  Code for data cleaning and splitting
# * Last modified :
# * *******************************************************
# * Filename      : data_processing
import pandas as pd
import pickle
import sys
import os


def loadfile(filename):
    """
    load a file, return a generator.
    """
    with open(filename, "r", encoding="ISO-8859-1") as fp:
        for i, line in enumerate(fp):
            yield line.strip("\r\n")
            if i % 10000 == 0 and i != 0:
                print("loading %s(%s)" % (filename, i), file=sys.stderr)
        print("load %s succeed" % filename, file=sys.stderr)


def gen_movies_dict(movies_data_path):
    """
    Generate movie data
    """
    movies_data = {}
    for line in loadfile(movies_data_path):
        MovieID, Title, Genres = line.strip().split("::")
        movies_data[MovieID] = (Title, Genres)
    return movies_data


def gen_users_dict(users_data_path):
    """
    Generate user data
    """
    users_data = {}
    for line in loadfile(users_data_path):
        UserID, Gender, Age, Occupation, Zip_code = line.strip().split("::")
        users_data[UserID] = (Gender, Age, Occupation)
    return users_data


def dump_user_item(data, path):
    with open(path, "wb") as meta:
        pickle.dump(data, meta)


def gen_sample_2_file(result_path):
    """
    Generate sample data and store in file
    """
    movies_data_path = "data_resource/origin_data/ml-1m/movies.dat"
    users_data_path = "data_resource/origin_data/ml-1m/users.dat"
    ratings_data_path = "data_resource/origin_data/ml-1m/ratings.dat"

    train_result_path = result_path + "train_sample"
    test_result_path = result_path + "test_sample"
    if os.path.exists(train_result_path):
        os.remove(train_result_path)
    if os.path.exists(test_result_path):
        os.remove(test_result_path)

    users_data = gen_users_dict(users_data_path)
    movies_data = gen_movies_dict(movies_data_path)

    dump_user_item(users_data, "./data_resource/pickl_data/all_user.data")
    dump_user_item(movies_data, "./data_resource/pickl_data/all_item.data")

    test_sample = dict()

    fo_train = open(train_result_path, "a", errors="ignore")
    for line in loadfile(ratings_data_path):
        UserID, MovieID, Rating, Timestamp = line.strip().split("::")
        Gender, Age, Occupation = users_data[UserID]
        Title, Genres = movies_data[MovieID]

        result = (
            UserID
            + "\t"
            + Gender
            + "\t"
            + Age
            + "\t"
            + Occupation
            + "\t"
            + MovieID
            + "\t"
            + Title
            + "\t"
            + Genres
            + "\t"
            + Rating
        )
        if UserID not in test_sample:
            test_sample[UserID] = (result, Timestamp)
        else:
            test_sample_result, test_sample_time = test_sample[UserID]
            if int(Timestamp) > int(test_sample_time):
                test_sample[UserID] = (result, Timestamp)
                fo_train.writelines(str(test_sample_result) + "\n")
            else:
                fo_train.writelines(str(result) + "\n")
    fo_train.close()
    fo_test = open(test_result_path, "a", errors="ignore")
    for k, v in test_sample.items():
        result_test, _ = v
        fo_test.writelines(str(result_test) + "\n")
    fo_test.close()


if __name__ == "__main__":
    """
    主函数入口
    Main function starts from here
    """

    result_path = "data_resource/sample_data/"
    gen_sample_2_file(result_path)
