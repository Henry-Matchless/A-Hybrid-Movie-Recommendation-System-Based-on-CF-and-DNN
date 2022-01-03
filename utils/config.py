# 这里面存所有的超参 
# It contains all the super-parameters.

model_type = "ncf"  # dnn  mf  ncf
learning_rate = 0.01
dropout = 0.0
epoch_num = 2
batch_size = 1024
batch_size_eval = 2048
dim_id = 64
dim_other = 8
loss = "log_loss"
eval_sample_count = 7000

train_path = "data_resource/sample_data/sample_train"
test_path = "data_resource/sample_data/sample_test"

model_path = "data_resource/savemodel"
checkpoint_path = "data_resource/checkpoint"
vocabulary_file_path = "data_resource/origin_data/vocabulary_file"
all_column_num = 8
all_slotid = ["0", "1", "2", "3", "4", "6"]  # 作为特征的index Index as a feature
user_slotid = ["0", "1", "2", "3"]
movie_slotid = ["4", "6"]
uid_nums = ("0", 6040)
itemid_nums = ("4", 3952)
slotid_name = {
    "0": "UserID",
    "1": "Gender",
    "2": "Age",
    "3": "Occupation",
    "4": "MovieID",
    "5": "Title",
    "6": "Genres",
}
multi_value_index = ["6"]

# 作为label的index index of label
label_index = 7

# label区分正负的阈值
# Label threshold for distinguishing positive and negative
label_threshold = 3
predict_threshold = 0.5


# dnn 模型的一些参数 
# Some parameters of dnn model
layer_nodes = [128, 64]

# 矩阵分解MF 参数
# MF parameter
reg_gamma = 0.01
is_reg = True
