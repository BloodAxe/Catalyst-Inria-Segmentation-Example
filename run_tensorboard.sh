#!/usr/bin/env bash
source activate tb
export CUDA_VISIBLE_DEVICES=
tensorboard --logdir runs --host 0.0.0.0 --port 5555