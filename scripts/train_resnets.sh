export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"

python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py --fp16 -w 8\
  -b 16 -m resnet101_unet64 --size 512\
  -o RAdam -lr 3e-4 -wd 1e-5  -a hard\
  -e 50 -s cos2\
  --criterion bce 1\
  --criterion dice 1\
  --train-mode tiles --show
