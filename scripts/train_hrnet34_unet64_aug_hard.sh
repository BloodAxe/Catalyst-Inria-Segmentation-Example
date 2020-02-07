python fit_predict.py\
  -dd "/home/bloodaxe/data/AerialImageDataset"\
  -m hrnet34_unet64\
  -a hard\
  -b 48\
  -o RAdam\
  -w 24\
  --fp16\
  -e 100\
  -s cos\
  -lr 1e-3\
  -wd 1e-6\
  --show\
  --seed 123\
  -l bce 1\
  -v

