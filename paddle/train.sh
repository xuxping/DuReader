#!/usr/bin/env bash
DPATH=../data/preprocessed

python run.py --prepare
python run.py --train --batch_size=32 --optim=adam --epochs=10 --log_interval=100 --train_embed=False


# demo
python run.py --prepare --demoo
python run.py --train --demo --batch_size=32 --epochs=4 --log_interval=10
