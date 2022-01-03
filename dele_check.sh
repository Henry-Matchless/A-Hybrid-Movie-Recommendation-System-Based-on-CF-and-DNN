#!/bin/bash
deleteDStore(){
    rm -rf ./.DS_Store
    echo === delete ./.DS_Store
    for file in ./*
    do
        if test -d $file
        then
            echo === delete $file/.DS_Store
            rm -rf $file/.DS_Store
        fi
    done
}

rm -rf data_resource/checkpoint/* data_resource/savemodel/*
rm -rf utils/__pycache__ src/__pycache__ model/__pycache__ utils/.DS_Store src/.DS_Store model/.DS_Store
echo ======== Curdir : $(cd "$(dirname "$0")";pwd) ========

deleteDStore
cd data_resource
deleteDStore

echo "======== Delete checkpoint and savemodel cache succeed!!! ========"
