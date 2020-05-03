@call c:\Anaconda3\Scripts\activate.bat pytorch14
python fit_predict.py -m densenet121_unet32  --train-mode tiles --show --fp16 -b 32 -w 8 --size 512 -s cos -o RAdam -a hard -lr 1e-4 -e 100 -d 0.1 --criterion bce 1 -v
REM python fit_predict.py -m densenet161_unet32  --train-mode tiles --show --fp16 -b 32 -w 8 --size 512 -s cos -o RAdam -a hard -lr 1e-4 -e 100 -d 0.1 --criterion bce 1 -v
REM python fit_predict.py -m densenet169_unet32  --train-mode tiles --show --fp16 -b 32 -w 8 --size 512 -s cos -o RAdam -a hard -lr 1e-4 -e 100 -d 0.1 --criterion bce 1 -v
REM python fit_predict.py -m densenet201_unet32  --train-mode tiles --show --fp16 -b 32 -w 8 --size 512 -s cos -o RAdam -a hard -lr 1e-4 -e 100 -d 0.1 --criterion bce 1 -v

