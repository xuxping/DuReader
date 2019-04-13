# -*- coding: utf8 -*-
import io
import json
import logging
import os
import sys

from gensim.models import KeyedVectors
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__file__)


def extract_text_from_file():
    def _read(file_path):
        with io.open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                sample = json.loads(line.strip())
                for doc in sample['documents']:
                    print(' '.join(doc['segmented_title']))
                    for paragraph in doc['segmented_paragraphs']:
                        print(' '.join(paragraph))

    all_file_path = ['../data/preprocessed/trainset/search.train.json',
                     '../data/preprocessed/trainset/zhidao.train.json',
                     '../data/preprocessed/devset/search.dev.json',
                     '../data/preprocessed/devset/zhidao.dev.json']

    for file_path in all_file_path:
        _read(file_path)


SAVE_WORD2VEC_FILE = '../data/vocab/vocab.wv.txt'


def train_wv():
    source = '../data/corpus.txt'
    logger.info(source)

    if not os.path.exists(source):
        sys.exit(-1)
    sentence = word2vec.LineSentence(source)
    model = word2vec.Word2Vec(sentence, size=200)

    model.wv.save_word2vec_format(SAVE_WORD2VEC_FILE)


def use_wv():
    model = KeyedVectors.load_word2vec_format(SAVE_WORD2VEC_FILE)
    for key in model.wv.similar_by_word(u'习近平', topn=10):
        print(key[0], key[1])


if __name__ == '__main__':
    train_wv()
    # extract_text_from_file()
