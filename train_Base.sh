rm -r ../trained
mkdir ../trained
CUDA_VISIBLE_DEVICES=0 python main.py --cache --train_Base
