export INRIA_DATA_DIR="/home/bloodaxe/datasets/AerialImageDataset"
python fit_predict.py -m resnet18_unet32  --train-mode tiles --show --fp16 -b 128 -w 25 --size 512 -s cosrd -o RAdam -a hard -lr 3e-4 -e 100 -d 0.1 --criterion bce 1 -v
python fit_predict.py -m resnet34_unet32  --train-mode tiles --show --fp16 -b 128 -w 25 --size 512 -s cosrd -o RAdam -a hard -lr 3e-4 -e 100 -d 0.1 --criterion bce 1 -v
python fit_predict.py -m resnet50_unet32  --train-mode tiles --show --fp16 -b 96 -w 25 --size 512 -s cosrd -o RAdam -a hard -lr 3e-4 -e 100 -d 0.1 --criterion bce 1 -v
python fit_predict.py -m resnet101_unet32  --train-mode tiles --show --fp16 -b 64 -w 25 --size 512 -s cosrd -o RAdam -a hard -lr 3e-4 -e 100 -d 0.1 --criterion bce 1 -v
python fit_predict.py -m resnet152_unet32 --train-mode tiles --fp16 -b 48 -w 16 --size 512 -s cosrd -o RAdam -a hard -lr 3e-4 -e 100 -d 0.1 --criterion bce 1 -v

