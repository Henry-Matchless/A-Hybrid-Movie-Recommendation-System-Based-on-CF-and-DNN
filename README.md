#  ***Recommendation Algorithms -- Neural Matrix Factorization, Matrix Factorization, DNN***
## 1. Environment
+  It is recommended to use virtual environment for environmental isolation.
+  Python 3.6 version
+  Installation based on anaconda virtual environment
```shell
conda create -n your_envname python=3.6
```
+ Switching virtual environment
```shell
activate your_envname
```
+ All the packages of this project have been put in `requirements.txt`.
+ The installation command is as follows
```shell
pip install -r requirements.txt
```
## 2. Data processing
+ The data processing code path is as follows
```shell
data_resource/data_processing.py
```
## 3. The entrance of Models
```shell
src/main.py
```
+ The main function is the entrance of the model, where the configuration of the model is in `utils／config.py`.
+ `dump_embeding.py` is used to extract the user vector and item vector in ckpt.
+ `predicted.py` makes recommendations according to vector distance.


> Attention! When entering from the `main.py` entrance, the files in `data _ resource \ checkpoint` and the models saved in `data _ resource \ savemodel` should be manually deleted after each run and the next model run.

> Attention! Conda virtual environment can use vpn pip but not vpn.
## 4. Introduction to data set
###      Ml-1M dataset field
+ ratings
`UserID::MovieID::Rating::Timestamp`
+ users
`UserID::Gender::Age::Occupation::Zip-code`
+ movies
`MovieID::Title::Genres`
## 5. Code for Item-based CF and User-based CF
+ The code path is located in `collaborative_filtering\itemcf.py` and `collaborative_filtering\usercf.py`.
```shell
python collaborative_filtering\itemcf.py
python collaborative_filtering\usercf.py
```
## 6. Data and Plots
+ All the Data and plots' codes is located in `data_resource\summary` file.
## **<u>For more convenience, you can use the automated bash script to run the model automatically, which is located under `run.sh`.It can help you automatically delete some files and rerun multiple customized Epoch.</u>**
```shell
python run.sh
```
