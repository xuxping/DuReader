## 采用迭代器改进数据加载方式
```

              total        used        free      shared  buff/cache   available
Mem:            31G        2.2G         14G        121M         14G         28G
Swap:            0B          0B          0B
```

## 测试环境
ubuntu 18.4 LTS
GTX 1080Ti, 显存11G
内存32G

## 词向量预训练
采用gensim在训练集和测试集上训练词向量，维度200维，训练5轮


## 训练参数
```
python run.py --prepare
python run.py --train --batch_size=32 --optim=adam --epochs=10 --log_interval=100
```
训练参数
```
    batch_size=32,
    demo=False,
    dev_interval=-1,
    devset=['../data/preprocessed/devset/search.dev.json',
    '../data/preprocessed/devset/zhidao.dev.json'],
    doc_num=5,
    drop_rate=0.1,
    embed_size=200,
    enable_ce=False,
    epochs=3,
    evaluate=False,
    hidden_size=150,
    learning_rate=0.001,
    load_dir='',
    log_interval=100,
    log_path='./logs',
    max_a_len=200,
    max_p_len=500,
    max_p_num=5,
    max_q_len=60,
    optim='adam',
    para_print=False,
    predict=False,
    prepare=False,
    random_seed=123,
    result_dir='../data/results/',
    result_name='test_result',
    save_dir='../data/models',
    save_interval=1,
    start_epoch=1,
    testset=['../data/preprocessed/test1set/search.test1.json',
    '../data/preprocessed/test1set/zhidao.test1.json'],
    train=True,
    train_embed=True,
    trainset=['../data/preprocessed/trainset/search.train.json',
    '../data/preprocessed/trainset/zhidao.train.json'],
    use_gpu=True,
    vocab_dir='../data/vocab',
    weight_decay=0.0001
```


## nvidia-smi
Fri Apr 12 13:25:02 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:0F:00.0 Off |                  N/A |
| 34%   58C    P2   101W / 250W |  10235MiB / 11178MiB |     28%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      2110      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      2243      G   /usr/bin/gnome-shell                          64MiB |
|    0      2610      G   /usr/bin/vlc                                   6MiB |
|    0      4249      C   python                                     10133MiB |
+-----------------------------------------------------------------------------+
