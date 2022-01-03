#!/bin/bash

clear
sh dele_check.sh
python src/main.py
tensorboard --logdir data_resource/checkpoint