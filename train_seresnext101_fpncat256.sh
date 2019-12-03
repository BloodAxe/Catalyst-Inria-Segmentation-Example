#!/usr/bin/env bash

#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a light  -lr 1e-3 -e 30 -d 0.5  -x seresnext101_fpncat256_light
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a medium -lr 3e-4 -e 30 -d 0.1  -x seresnext101_fpncat256_medium -c runs/seresnext101_fpncat256_light/checkpoints/best.pth
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a hard   -lr 1e-4 -e 30 -d 0.05 -x seresnext101_fpncat256_hard  --run-mode fit -c runs/seresnext101_fpncat256_medium/checkpoints/best.pth
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --tta d4 --size 512 -s cos -o RAdam -a light  -lr 1e-4 -e 50 -d 0.05 -x seresnext101_fpncat256_light2  --run-mode fit -c runs/seresnext101_fpncat256_hard/checkpoints/best.pth

#python fit_predict.py -m seresnext50_unet64     --fp16 -b 64 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cosr -o RAdam -a medium -lr 1e-4 -e 50 -d 0.05 -c runs/Nov22_18_17_seresnext50_unet64_fp16/main/checkpoints/train.35.pth -v --show -l soft_bce 1 -l jaccard 0.2 -tm tiles
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cosr -o RAdam -a medium -lr 1e-4 -e 50 -d 0.05 -c runs/seresnext101_fpncat256_light2/checkpoints/best.pth -v --show  -l focal 1 -tm tiles
#python fit_predict.py -m seresnext101_fpncat256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cosr -o RAdam -a safe -lr 1e-4 -e 50 -d 0.05 -c runs/Nov23_09_18_seresnext101_fpncat256_fp16/main/checkpoints/best.pth -v --show  -l focal 1
python fit_predict.py -m seresnext101_deeplab256 --fp16 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cosr -o RAdam -a medium -lr 1e-4 -e 75 -d 0.1 --transfer runs/Nov23_17_14_seresnext101_deeplab256_fp16/main/checkpoints/best.pth -v  -l focal 1
python fit_predict.py -m seresnext101_rfpncat256 --fp16 -b 24 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cosr -o RAdam -a medium -lr 1e-4 -e 75 -d 0.1 --transfer runs/Nov23_09_18_seresnext101_fpncat256_fp16/main/checkpoints/best.pth -v  -l focal 1


python fit_predict.py -m hrnet48 -b 32 -w 16 -dd /home/bloodaxe/data/AerialImageDataset --size 512 -s cos -o RAdam -a medium -lr 1e-3 -e 200 -d 0.1 -v -l bce 1
