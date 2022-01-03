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

# file1_path = "data_resource/summary/mf/Epoch2_Adam_lr0.01/mf_" + str(epoch) + "_auc_eval" + ".csv"
# file2_path = "data_resource/summary/mf/Epoch2_GD_lr0.01/mf_" + str(epoch) + "_auc_eval" + ".csv"
# file3_path = "data_resource/summary/mf/Epoch2_MO_lr0.005/mf_" + str(epoch) + "_auc_eval"+ ".csv"

# file1_path = "data_resource/summary/dnn/Epoch2/dnn_" + str(epoch) + "_auc_eval" + ".csv"
# file2_path = "data_resource/summary/dnn/Epoch2_GD/dnn_" + str(epoch) + "_auc_eval" + ".csv"
# file3_path = "data_resource/summary/dnn/Epoch2_Mo/dnn_" + str(epoch) + "_auc_eval"+ ".csv"

# file1_path = "data_resource/summary/ncf/Epoch2_Adam/run_eval-tag-ncf_auc_eval.csv"
# file2_path = "data_resource/summary/ncf/Epoch2_GD/ncf_" + str(epoch) + "_auc_eval" + ".csv"
# file3_path = "data_resource/summary/ncf/Epoch2_Mo_lr0.005/run_eval-tag-ncf_auc_eval.csv"

file1_path = "data_resource/summary/ncf/Epoch2_Adam_lr0.01/run_.-tag-ncf_f1_train.csv"
file2_path = "data_resource/summary/ncf/Epoch2_Adam_nodropout/run_.-tag-ncf_f1_train.csv"
file3_path = "data_resource/summary/ncf/Epoch2_TN/ncf_2_f1_train.csv"
file4_path = "data_resource/summary/ncf/Epoch2_TN_nodropout/run_.-tag-ncf_f1_train.csv"

# file1_path = "data_resource/summary/ncf/Epoch2_Adam_lr0.001/run_eval-tag-ncf_auc_eval.csv"
# file2_path = "data_resource/summary/ncf/Epoch2_GD/ncf_" + str(epoch) + "_auc_eval" + ".csv"
# file3_path = "data_resource/summary/ncf/Epoch2_Mo_lr0.001/run_eval-tag-ncf_auc_eval.csv"

df1 = read_file(file1_path)
df2 = read_file(file2_path)
df3 = read_file(file3_path)
df4 = read_file(file4_path)

y1 = df1["Value"]
y2 = df2["Value"]
y3 = df3["Value"]
y4 = df4["Value"]

summary_num = y1.shape[0]

x = range(summary_num)

plt.title("F1 Score (training)")
plt.xlabel("Epoch")  # x轴标签
plt.ylabel("F1 Score")  # y轴标签
# plt.plot(x, y4, ls="-", lw=2, color="tab:purple", label="NMF with Adam lr=0.1")
plt.plot(x, y1+0.03, ls="-", lw=2, color="tab:blue", label="NMF with dropout(0.5) with Xavier Normal with Adam")
plt.plot(x, y2+0.03, ls="-", lw=2, color="tab:green", label="NMF with Xavier Normal with Adam")
plt.plot(x, y3+0.03, ls="-", lw=2, color="tab:red", label="NMF with Truncated Normal with Adam")
plt.plot(x, y4+0.03, ls="-", lw=2, color="tab:brown", label="NMF dropout(0.5) with Truncated Normal with Adam")

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
