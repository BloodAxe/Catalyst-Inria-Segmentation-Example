export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"

python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py -w 6 --fp16 -v\
  -b 12 -m b6_unet32_s2\
  --train-mode tiles -b 8 --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 100\
  --criterion bce 1 --criterion dice 1