#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def read_file(filename):
    """
    读取 csv 文件
    filename:文件路径
    """
    dataframe = pd.read_csv(filename)

    return dataframe


epoch = 2

file1_path = "data_resource/summary/dnn/Epoch2/dnn_" + str(epoch) + "_loss" + ".csv"
file2_path = "data_resource/summary/ncf/Epoch2/ncf_" + str(epoch) + "_loss" + ".csv"
file3_path = "data_resource/summary/mf/Epoch2/mf_" + str(epoch) + "_loss"+ ".csv"

df1 = read_file(file1_path)
df2 = read_file(file2_path)
df3 = read_file(file3_path)

y1 = df1["Value"]
y2 = df2["Value"]
y3 = df3["Value"]

summary_num = y1.shape[0]

x = range(summary_num)

plt.title("Train Loss")
plt.xlabel("Epoch")  # x轴标签
plt.ylabel("LOSS")  # y轴标签
plt.plot(x, y1, ls="-", lw=2, color="tab:blue", label="DNN")
plt.plot(x, y2, ls="-", lw=2, color="tab:red", label="NMF")
plt.plot(x, y3, ls="-", lw=2, color="tab:green", label="MF")

epoch_range = {
    "1": ("0", "0.25", "0.5", "0.75", "1"),
    "2": ("0", "0.5", "1", "1.5", "2"),
    "3": ("0", "1", "2", "3"),
    "4": ("0", "1", "2", "3", "4"),
    "5": ("0", "1", "2", "3", "4", "5"),
}
label_num = len(epoch_range[str(epoch)]) - 1
ticks_range = np.arange(0, summary_num + 1, int(summary_num / label_num))


plt.xticks(
    ticks=ticks_range,
    labels=epoch_range[str(epoch)],
)


plt.legend()
plt.savefig("data_resource/summary/epoch_" + str(epoch) + ".png")
plt.show()
