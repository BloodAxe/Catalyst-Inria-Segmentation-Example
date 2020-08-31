#export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"
#python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py -w 6 --fp16 -v\
#  -b 6 -m b6_unet32_s2\
#  --train-mode tiles -b 8 --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 100\
#  --criterion bce 1 --criterion dice 1

export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"
python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py -w 6 --fp16 -v\
  -b 6 -m b6_unet32_s2_tc\
  --train-mode tiles --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 50 --seed 555\
  --criterion bce 1 --criterion dice 1 --transfer /home/bloodaxe/develop/Catalyst-Inria-Segmentation-Example/runs/200829_17_44_b6_unet32_s2_fp16_local_rank_0/main/checkpoints_optimized_jaccard/best.pth

export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"
python -m torch.distributed.launch --nproc_per_node=4 fit_predict.py -w 6 --fp16 -v\
  -b 6 -m b6_unet32_s2_rdtc\
  --train-mode tiles --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 50 --seed 555\
  --criterion bce 1 --criterion dice 1\
  -l2 bce 0.75\
  -l4 bce 0.5\
  -l8 bce 0.25\
  -l16 bce 0.125\
  --transfer /home/bloodaxe/develop/Catalyst-Inria-Segmentation-Example/runs/200830_23_50_b6_unet32_s2_bi_fp16_local_rank_0/200830_23_50_b6_unet32_s2_bi_fp16_local_rank_0.pth