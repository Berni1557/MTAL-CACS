#!/bin/sh

python cacs_predict.py \
    -m ../model/model.pt \
    -d ../data \
    -p ../prediction \
    -gpu cuda \
