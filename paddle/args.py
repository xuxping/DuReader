#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # set run mode
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--demo', action='store_true', help='use demo data path')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')

    # set model hyper
    parser.add_argument("--embed_size", type=int, default=200,
                        help="The dimension of embedding table. (default: %(default)d)")

    parser.add_argument("--hidden_size", type=int, default=100,
                        help="The dimension of embedding table. (default: %(default)d)")

    parser.add_argument("--train_embed", type=distutils.util.strtobool, default=True,
                        help="et embedding trainable. (default: %(default)d)")

    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate used to train the model. (default: %(default)f)")
    parser.add_argument('--optim', default='adam', help='optimizer type')
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay. (default: %(default)f)")
    parser.add_argument('--drop_rate', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The sequence number of a mini-batch data. (default: %(default)d)")

    # set document
    parser.add_argument('--max_p_num', type=int, default=5)
    parser.add_argument('--max_a_len', type=int, default=200)
    parser.add_argument('--max_p_len', type=int, default=500)
    parser.add_argument('--max_q_len', type=int, default=60)
    parser.add_argument('--doc_num', type=int, default=5)

    # set run epochs and gpu
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="resume from load_dir (default: %(default)d)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="The number epochs to train. (default: %(default)d)")
    parser.add_argument("--use_gpu", type=distutils.util.strtobool, default=True,
                        help="Whether to use gpu. (default: %(default)d)")

    # set path
    parser.add_argument('--vocab_dir', default='../data/vocab', help='vocabulary')
    parser.add_argument("--save_dir", type=str, default="../data/models",
                        help="Specify the path to save trained models.")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save the trained model every n passes. (default: %(default)d)")
    parser.add_argument("--load_dir", type=str, default="",
                        help="Specify the path to load trained models.")
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='path of the log file. If not set, logs are printed to console')
    parser.add_argument('--result_dir', default='../data/results/',
                        help='the dir to output the results')
    parser.add_argument('--result_name', default='test_result',
                        help='the file name of the predicted results')

    # set dataset
    parser.add_argument('--trainset', nargs='+',
                        default=['../data/extracted/trainset/search.train.json',
                                 '../data/extracted/trainset/zhidao.train.json'],
                        help='train dataset')
    parser.add_argument('--devset', nargs='+',
                        # default=[],
                        default=['../data/extracted/devset/search.dev.json',
                                 '../data/extracted/devset/zhidao.dev.json'],
                        help='dev dataset')
    parser.add_argument('--testset', nargs='+',
                        default=['../data/preprocessed/test1set/search.test1.json',
                                 '../data/preprocessed/test1set/zhidao.test1.json'],
                        help='test dataset')

    parser.add_argument("--save_result", action='store_true',
                        help="If set, save dev/test result into result.json.")
    parser.add_argument("--enable_ce", action='store_true', default=True,
                        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument('--para_print', action='store_true', help="Print debug info")
    parser.add_argument("--dev_interval", type=int, default=-1,
                        help="evaluate on dev set loss every n batches. (default: %(default)d)")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="log the train loss every n batches. (default: %(default)d)")
    args = parser.parse_args()

    if args.demo:
        args.trainset = ['../data/demo/trainset/search.train.json']
        args.devset = ['../data/demo/devset/search.dev.json']
        args.testset = ['../data/demo/testset/search.test.json']

    return args
