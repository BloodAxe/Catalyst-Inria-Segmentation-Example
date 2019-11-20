#!/usr/bin/env bash

#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a light  -lr 1e-3 -e 30 -d 0.5  -x seresnext101_fpncat256_light
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a medium -lr 3e-4 -e 30 -d 0.1  -x seresnext101_fpncat256_medium -c runs/seresnext101_fpncat256_light/checkpoints/best.pth
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a hard   -lr 1e-4 -e 30 -d 0.05 -x seresnext101_fpncat256_hard  --run-mode fit -c runs/seresnext101_fpncat256_medium/checkpoints/best.pth
python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a light  -lr 1e-4 -e 50 -d 0.05 -x seresnext101_fpncat256_light2  --run-mode fit -c runs/seresnext101_fpncat256_hard/checkpoints/best.pth
