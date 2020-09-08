export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"
python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py -w 6 --fp16 -v\
  -b 10 -m mxxl_unet32_s1\
  --train-mode tiles --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 100 --seed 555\
  --criterion bce 1 --criterion dice 1
