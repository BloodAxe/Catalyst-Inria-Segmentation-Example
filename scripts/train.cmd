@call c:\Anaconda3\Scripts\activate.bat pytorch14
REM --fp16 -m b4_unet32 --train-mode tiles --show -b 10 -w 4 --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 100 --criterion bce 1 --criterion dice 0.1 -v -dd d:\datasets\AerialImageDataset

python fit_predict.py --fp16 -m b4_unet32_s2 --train-mode tiles --show -b 8 -w 4 --size 512 -s cos -o RAdam -a hard -lr 3e-4 -e 100 --criterion bce 1 --criterion dice 0.1 -v -dd d:\datasets\AerialImageDataset
