# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""
import io
import json
import logging
import mmap
import random
from collections import Counter

import numpy as np

from vocab import Vocab


class MmapFile(object):
    """
    Use mmap to improve read speed
    """
    def __init__(self, datafile):
        headerfile = datafile + '.header'
        self.offsetdict = {}
        for line in open(headerfile, 'r'):
            key, val_pos, val_len = line.split('\t')
            self.offsetdict[key] = (int(val_pos), int(val_len))

        # self.fp = open(datafile, 'rb')
        with open(datafile, 'rb') as fp:
            self.m = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    def getvalue(self, key):
        p = self.offsetdict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:(val_pos + val_len)]

    def getdictlen(self):
        return len(self.offsetdict)


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self,
                 max_p_num,
                 max_p_len,
                 max_q_len,
                 train_files=[],
                 dev_files=[],
                 test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files

        self.train_set, self.dev_set, self.test_set = [], [], []
        self.vocab = None

    def path_reader(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        Yields: a sample
        """
        with io.open(data_path, 'r', encoding='utf-8') as fin:

            for lidx, line in enumerate(fin):

                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append({
                            'passage_tokens':
                                doc['segmented_paragraphs'][most_related_para],
                            'is_selected': doc['is_selected']
                        })
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(
                                para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(
                                    correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question,
                                               len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({
                            'passage_tokens': fake_passage_tokens
                        })

                # self.convert_to_ids(sample)
                yield sample

    def path_reader2(self, reader, batch_indices, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        Yields: a sample
        """
        data_set = []
        for idx in batch_indices:
            line = reader.getvalue(str(idx))
            if not line:
                continue

            sample = json.loads(bytes.decode(line.strip()))
            if train:
                if len(sample['answer_spans']) == 0:
                    continue
                if sample['answer_spans'][0][1] >= self.max_p_len:
                    continue

            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']

            sample['question_tokens'] = sample['segmented_question']

            sample['passages'] = []
            for d_idx, doc in enumerate(sample['documents']):
                if train:
                    most_related_para = doc['most_related_para']
                    sample['passages'].append({
                        'passage_tokens':
                            doc['segmented_paragraphs'][most_related_para],
                        'is_selected': doc['is_selected']
                    })
                else:
                    para_infos = []
                    for para_tokens in doc['segmented_paragraphs']:
                        question_tokens = sample['segmented_question']
                        common_with_question = Counter(
                            para_tokens) & Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(
                                correct_preds) / len(question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question,
                                           len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    fake_passage_tokens = []
                    for para_info in para_infos[:1]:
                        fake_passage_tokens += para_info[0]
                    sample['passages'].append({
                        'passage_tokens': fake_passage_tokens
                    })
            data_set.append(sample)
        return data_set

    # def _one_mini_batch(self, data, indices, pad_id):
    def _one_mini_batch(self, batch_list, shuffle=True):
        """
        Get one mini batch
        Args:
            batch_list: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """

        batch_data = {
            'raw_data': [],
            'question_token_ids': [],
            'question_length': [],
            'passage_token_ids': [],
            'passage_length': [],
            'start_id': [],
            'end_id': [],
            'passage_num': []
        }
        if shuffle:
            random.shuffle(batch_list)

        for sample in batch_list:
            self.convert_to_ids(sample)
            batch_data['raw_data'].append(sample)

        max_passage_num = max(
            [len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            count = 0
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    count += 1
                    batch_data['question_token_ids'].append(sample[
                                                                'question_token_ids'][0:self.max_q_len])
                    batch_data['question_length'].append(
                        min(len(sample['question_token_ids']), self.max_q_len))
                    passage_token_ids = sample['passages'][pidx][
                                            'passage_token_ids'][0:self.max_p_len]
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(
                        min(len(passage_token_ids), self.max_p_len))

            # record the start passage index of current sample
            passade_idx_offset = sum(batch_data['passage_num'])
            batch_data['passage_num'].append(count)
            gold_passage_offset = 0
            if 'answer_passages' in sample and len(sample['answer_passages']) and \
                    sample['answer_passages'][0] < len(sample['documents']):
                for i in range(sample['answer_passages'][0]):
                    gold_passage_offset += len(batch_data['passage_token_ids'][
                                                   passade_idx_offset + i])

                start_id = min(sample['answer_spans'][0][0], self.max_p_len)
                end_id = min(sample['answer_spans'][0][1], self.max_p_len)
                batch_data['start_id'].append(gold_passage_offset + start_id)
                batch_data['end_id'].append(gold_passage_offset + end_id)
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)

        return batch_data

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            # data_set = self.train_set + self.dev_set + self.test_set
            data_path_set = self.train_files + self.dev_files + self.test_files
            raise ValueError('invalid set_name')
        elif set_name == 'train':
            # data_set = self.train_set
            data_path_set = self.train_files
        elif set_name == 'dev':
            # data_set = self.dev_set
            data_path_set = self.dev_files
        elif set_name == 'test':
            data_path_set = self.test_files
            # data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(
                set_name))
        is_train = True if set_name == 'train' else False
        if data_path_set is not None:
            for data_path in data_path_set:
                for sample in self.path_reader(data_path, train=is_train):
                    for token in sample['question_tokens']:
                        yield token
                    for passage in sample['passages']:
                        for token in passage['passage_tokens']:
                            yield token

    def set_vocab(self, vocab):
        if not isinstance(vocab, Vocab):
            raise ValueError('is not instance of Vocab')

        self.vocab = vocab

    def convert_to_ids(self, sample):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """

        sample['question_token_ids'] = self.vocab.convert_to_ids(sample[
                                                                     'question_tokens'])
        for passage in sample['passages']:
            passage['passage_token_ids'] = self.vocab.convert_to_ids(passage[
                                                                         'passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            # data = self.train_set
            path_set = self.train_files
        elif set_name == 'dev':
            # data = self.dev_set
            path_set = self.dev_files
        elif set_name == 'test':
            # data = self.test_set
            path_set = self.test_files
        else:
            raise NotImplementedError('No data set named as {}'.format(
                set_name))

        is_train = True if set_name == 'train' else False

        # 1, use mmap
        for data_path in path_set:
            print('load data from {}'.format(data_path))
            reader = MmapFile(data_path)

            data_size = reader.getdictlen()
            indices = np.arange(data_size)

            if shuffle:
                np.random.shuffle(indices)

            for batch_start in np.arange(0, data_size, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                data = self.path_reader2(reader, batch_indices, train=is_train)
                yield self._one_mini_batch(data, shuffle=False)

        # 2、use file
        # for data_path in path_set:
        #     print('load data from {}'.format(data_path))
        # batch_list = []
        # for sample in self.path_reader(data_path, train=is_train):
        #     batch_list.append(sample)
        #     size = len(batch_list)
        #     if size == batch_size:
        #         batch_data = self._one_mini_batch(batch_list, shuffle=True)
        #         yield batch_data
        #         batch_list = []
        #
        # if len(batch_list) > 0:
        #     batch_data = self._one_mini_batch(batch_list, shuffle=True)
        #     yield batch_data
