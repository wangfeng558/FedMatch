import os
from typing import Any

import cv2
import time
import random
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import keras
from torchvision.datasets.utils import check_integrity
import pandas as pd
from config import *
from misc.utils import *
from torchvision import datasets, transforms
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types


class DataGenerator:

    def __init__(self, args):

        self.data: Any = []
        self.targets = []
        self.args = args
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task)
        self.shape = (77, 1)
        self.train_list = [
            'data_party0.csv',
            'data_party1.csv',
            'data_party2.csv',
            'data_party3.csv',
            'data_party4.csv',
            'data_party5.csv',
            'data_party6.csv',
            'data_party7.csv',
            'data_party8.csv',
            'data_party9.csv']

        self.test_list = ['test.csv']

        self.server_list = ['server.csv']


    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1

        self.TON_LOT()

        print(f'{self.args.task} done ({time.time() - start_time}s)')

    #生成npy数据

    def TON_LOT(self, train: bool = True):

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        for file_name in self.train_list:
            file_path = os.path.join(self.args.dataset_path, file_name)
            x, y = [], []
            x = pd.read_csv(file_path)
            y = x['Label']
            del x['Label']

            y_train = tf.keras.utils.to_categorical(y, len(self.labels))
            l_train = np.unique(y_train)
            self.save_task({
                # 506
                'x': x,
                'y': y_train,
                'labels': l_train,
                'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}'
            })

        for file_name in self.test_list:
            file_path = os.path.join(self.args.dataset_path, file_name)
            x, y = [], []
            x = pd.read_csv(file_path)
            y = x['Label']
            del x['Label']

            y_test = tf.keras.utils.to_categorical(y, len(self.labels))
            l_test = np.unique(y_test)
            self.save_task({
                # 506
                'x': x,
                'y': y_test,
                'labels': l_test,
                'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
            })
        for file_name in self.server_list:
            file_path = os.path.join(self.args.dataset_path, file_name)
            x, y = [], []
            x = pd.read_csv(file_path)
            y = x['Label']
            del x['Label']
            y_server = tf.keras.utils.to_categorical(y, len(self.labels))
            l_server = np.unique(y_server)
            self.save_task({
                # 506
                'x': x,
                'y': y_server,
                'labels': l_server,
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}'
            })

    def _check_integrity(self) -> bool:
        for fentry in (self.train_list + self.test_list):
            filename = fentry
            fpath = os.path.join(self.args.dataset_path, filename)
            if not os.path.isfile(fpath):
                return False
        return True

    def generate_task(self, x, y):
        x_train, y_train = self.split_train_test_valid(x, y)
        s, u = self.split_s_and_u(x_train, y_train)
        self.split_s(s)
        self.split_u(u)

    def split_train_test_valid(self, x, y):
        self.num_examples = len(x)
        self.num_train = self.num_examples - (self.args.num_test + self.args.num_valid)
        # test_num:2000 vaild = 2000
        self.num_test = self.args.num_test

        self.labels = np.unique(y)

        print(f'num_train : {self.num_train}  num_examples: {self.num_examples}  num_test: {self.num_test}')

        # train set
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train + self.num_test]
        y_test = y[self.num_train:self.num_train + self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        l_test = np.unique(y_test)
        self.save_task({
            # 506
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # valid set
        x_valid = x[self.num_train + self.num_test:]
        y_valid = y[self.num_train + self.num_test:]
        # 分成10类
        y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        l_valid = np.unique(y_valid)
        self.save_task({
            'x': x_valid,
            'y': y_valid,
            'labels': l_valid,
            'name': f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        return x_train, y_train

    def split_s_and_u(self, x, y):
        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class
        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients

        data_by_label = {}
        for label in self.labels:
            idx = np.where(y[:] == label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}
        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:self.num_s],
                'y': data['y'][:self.num_s]
            }
            u_by_label[label] = {
                'x': data['x'][self.num_s:],
                'y': data['y'][self.num_s:]
            }
            self.num_u += len(u_by_label[label]['x'])

        return s_by_label, u_by_label

    def split_s(self, s):
        if self.is_labels_at_server:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():
                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(y_labeled)
            })
        else:
            for cid in range(self.args.num_clients):
                x_labeled = []
                y_labeled = []
                for label, data in s.items():
                    start = self.args.num_labels_per_class * cid
                    end = self.args.num_labels_per_class * (cid + 1)
                    _x = data['x'][start:end]
                    _y = data['y'][start:end]
                    x_labeled = [*x_labeled, *_x]
                    y_labeled = [*y_labeled, *_y]
                x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_labeled)
                })

    def split_u(self, u):
        if self.is_imbalanced:
            ten_types_of_class_imbalanced_dist = [
                [0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15],  # type 0
                [0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03],  # type 1
                [0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03],  # type 2
                [0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02, 0.03],  # type 3
                [0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03, 0.02],  # type 4
                [0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03, 0.03],  # type 5
                [0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03, 0.03],  # type 6
                [0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15, 0.03],  # type 7
                [0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50, 0.15],  # type 8
                [0.15, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.15, 0.50],  # type 9
            ]
            labels = list(u.keys())
            num_u_per_client = int(self.num_u / self.args.num_clients)
            offset_per_label = {label: 0 for label in labels}
            for cid in range(self.args.num_clients):
                # batch-imbalanced
                x_unlabeled = []
                y_unlabeled = []
                dist_type = cid % len(labels)
                freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])
                frq = []
                for label, data in u.items():
                    num_instances = len(freqs[freqs == label])
                    frq.append(num_instances)
                    start = offset_per_label[label]
                    end = offset_per_label[label] + num_instances
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]]
                    offset_per_label[label] = end
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })
        else:
            # batch-iid
            for cid in range(self.args.num_clients):
                x_unlabeled = []
                y_unlabeled = []
                for label, data in u.items():
                    # print('>>> ', label, len(data['x']))
                    num_unlabels_per_class = int(len(data['x']) / self.args.num_clients)
                    start = num_unlabels_per_class * cid
                    end = num_unlabels_per_class * (cid + 1)
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]]
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })

    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")

    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]