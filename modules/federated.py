import os
import sys
import copy
import time
import random
import threading
import atexit
from queue import Queue
from typing import Any

import tensorflow as tf
from datetime import datetime

from misc.utils import *
from misc.logger import Logger
from data.loader import DataLoader
from modules.nets import NetModule
from modules.train import TrainModule
import queue as q


class ServerModule:

    queue: Queue[Any]

    def __init__(self, args, Client):

        self.args = args
        self.client = Client
        self.clients = {}
        self.threads = []
        self.updates = []
        self.task_names = []
        self.curr_round = -1
        self.limit_gpu_memory()
        self.logger = Logger(self.args)
        self.loader = DataLoader(self.args)
        self.net = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger)
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        atexit.register(self.atexit)
        self.queue = q.Queue()
        self.clients_round_weight = [100 for v in range(10)]
        self.basic_dataset = [
            [0.05, 0.58, 0.17, 0.17, 0.03, 0, 0, 0, 0],  # type 0
            [0.99, 0.1, 0, 0, 0, 0, 0, 0, 0],  # type 1
            [0.16, 0.80, 0.02, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001],  # type 2
            [0.999, 0.001, 0, 0, 0, 0, 0, 0, 0],  # type 3
            [0.006, 0.725, 0.157, 0.089, 0.021, 0, 0, 0, 0],  # type 4
            [0.031, 0.622, 0.133, 0.203, 0.008, 0, 0, 0, 0],  # type 5
            [0.016, 0.939, 0.044, 0, 0, 0, 0, 0, 0],  # type 6
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # type 7
            [0.048, 0.952, 0, 0, 0, 0, 0, 0, 0],  # type 8
            [0.046, 0.954, 0, 0, 0, 0, 0, 0, 0],  # type 9
        ]
        self.basic_entropy = [0.52041, 0, 0.28669, 0, 0.38890, 0.47291, 0.11976, 0.0002, 0.08794, 0.08511]
        self.basic_nums = [811504, 763518, 740117, 519806, 424531, 330956, 223092, 217737, 186891, 185932]

        self.balance_dataset = [
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 0
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 1
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 2
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 3
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 4
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 5
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 6
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 7
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 8
            [0.230, 0.230, 0.230, 0.230, 0.069, 0.0001, 0.0001, 0.0001, 0.0001],  # type 9
        ]
        self.balance_entropy = [0.7611, 0.7611, 0.7611, 0.7611, 0.7611, 0.7611, 0.7611, 0.7611, 0.7611, 0.7611]
        self.balance_nums = [43549, 43549, 43549, 43549, 43549, 43549, 43549, 43549, 43549, 43549]

        self.mix_dataset = [
            [0.206, 0.242, 0.242, 0.242, 0.065, 0, 0, 0, 0],  # type 0
            [0.234, 0.234, 0.234, 0.026, 0.256, 0.0001, 0.0001, 0.0001, 0.0001],  # type 1
            [0.038, 0.279, 0.279, 0.279, 0.125, 0, 0, 0, 0],  # type 2
            [0.143, 0.272, 0.272, 0.272, 0.040, 0, 0, 0, 0],  # type 3
        ]
        self.mix_entropy = [0.69858, 0.70266, 0.66218, 0.66888]
        self.mix_nums = [205946, 42679, 71748, 73446]


    def limit_gpu_memory(self):
        """ Limiting gpu memories

        Tensorflow tends to occupy all gpu memory. Specify memory size if needed at config.py.
        Please set at least 6 or 7 memory for runing safely (w/o memory overflows).
        """
        self.gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
        self.gpus = tf.config.list_physical_devices('GPU')
        if len(self.gpus) > 0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * self.args.gpu_mem)])

    def run(self):
        self.logger.print('server', 'server process has been started')
        self.load_data()
        self.build_network()
        self.net.init_state('server')
        self.net.set_init_params()
        self.train.init_state('server')
        self.train.set_details({
            'model': self.global_model,
            'loss_fn': self.loss_fn,
            'trainables': self.trainables,
            'num_epochs': self.args.num_epochs_server,
            'batch_size': self.args.batch_size_server,
        })
        self.create_clients()
        self.train_clients()

    def load_data(self):

        self.x_train, self.y_train, self.task_name = self.loader.get_s_server()

        self.x_valid, self.y_valid = self.loader.get_valid()
        self.x_test, self.y_test = self.loader.get_test()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid)
        self.train.set_task({
            'task_name': self.task_name,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            'x_test': self.x_test,
            'y_test': self.y_test,
        })

    def create_clients(self):
        args_copied = copy.deepcopy(self.args)
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
            gpu_ids_real = [int(gid) for gid in self.args.gpu.split(',')]
            cid_offset = 0
            self.logger.print('server', 'creating client processes on gpus ... ')
            print('creating client processes on gpus ...')
            for i, gpu_id in enumerate(gpu_ids):
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    self.clients[gpu_id] = self.client(gpu_id, args_copied)
        else:
            print('creating client processes on cpu ...')
            self.logger.print('server', 'creating client processes on cpu ... ')
            num_parallel = 10
            self.clients = {i: self.client(i, args_copied) for i in range(num_parallel)}

    def train_clients(self):
        start_time = time.time()
        self.threads = []
        self.updates = []
        self.connected_ids = np.arange(self.args.num_clients).tolist()
        # num_connected = int(round(self.args.num_clients*self.args.frac_clients))

        for curr_round in range(self.args.num_rounds * self.args.num_tasks):
            self.curr_round = curr_round

            #训练全局模型
            self.train_global_model()
            self.logger.print('server', f'training clients (round:{self.curr_round}, connected:{self.connected_ids})')
            self._train_clients()

            while (self.queue.qsize() < int(round(self.args.num_clients * self.args.frac_clients))):
                time.sleep(1)

        self.logger.print('server', 'all clients done')
        self.logger.print('server', 'server done. ({}s)'.format(time.time() - start_time))
        sys.exit()

    def aggregate(self, updates, sigma, nums, curr_round):
        return self.train.uniform_average(updates, sigma, nums, curr_round)


    def train_global_model(self):
        self.logger.print('server', 'training global_model')
        num_epochs = self.args.num_epochs_server_pretrain if self.curr_round == 0 else self.args.num_epochs_server
        self.train.train_global_model(self.curr_round, self.curr_round, num_epochs)

    def loss_fn(self, x, y):
        x = self.loader.scale(x)
        y_pred = self.global_model(x)
        # 交叉熵 损失函数
        loss = self.cross_entropy(y, y_pred) * self.args.lambda_s
        return y_pred, loss

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.logger.print('server', 'all client threads have been destroyed.')


########################################################################################
########################################################################################
########################################################################################

class ClientModule:

    def __init__(self, gid, args):
        """ Superclass for Client Module

        This module contains common client functions,
        such as loading data, training local model, switching states, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.state = {'gpu_id': gid}
        self.logger = Logger(self.args)
        self.loader = DataLoader(self.args)
        self.net = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger)

    def train_one_round(self, client_id, curr_round, weights=None, sigma=None, psi=None, slope=1):
        self.switch_state(client_id)
        if self.state['curr_task'] < 0:
            self.init_new_task()
        else:
            self.state['curr_lr'] = self.state['curr_lr'] / slope
            self.is_last_task = (self.state['curr_task'] == self.args.num_tasks - 1)
            self.is_last_round = (self.state['round_cnt'] % self.args.num_rounds == 0 and self.state['round_cnt'] != 0)
            self.is_last = self.is_last_task and self.is_last_round
            if self.is_last_round or self.train.state['early_stop']:
                if self.is_last_task:
                    # if self.train.state['early_stop']:
                    #     self.train.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
            else:
                self.load_data()
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        #######################################
        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            self._train_one_round(client_id, curr_round, sigma, psi)

        #######################################
        self.save_state()

        return (self.get_client_train_weights(), self.get_train_size(), self.state['client_id'],
                curr_round)

    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.net.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
        else:  # load_state
            self.net.load_state(client_id)
            self.train.load_state(client_id)
            self.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.check_pts, f'{client_id}_client.npy'))

    def init_state(self, client_id):
        self.state['client_id'] = client_id
        self.state['done'] = False
        self.state['curr_task'] = -1
        self.state['task_names'] = []
        self._init_state()

    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f'{client_id}_client.npy')

    def save_state(self):
        self.net.save_state()
        self.train.save_state()
        np_save(self.args.check_pts, f"{self.state['client_id']}_client.npy", self.state)

    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()

    def load_data(self):
        if self.args.scenario == 'labels-at-client':
            if 'simb' in self.args.task and self.state['curr_task'] > 0:
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
            else:
                self.x_labeled, self.y_labeled, task_name = \
                    self.loader.get_s_by_id(self.state['client_id'])
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        elif self.args.scenario == 'labels-at-server':
            self.x_labeled, self.y_labeled = None, None
            self.x_unlabeled, self.y_unlabeled, task_name = \
                self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        self.x_test, self.y_test = self.loader.get_test()
        self.x_valid, self.y_valid = self.loader.get_valid()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid)
        self.train.set_task({
            'task_name': task_name.replace('u_', ''),
            'x_labeled': self.x_labeled,
            'y_labeled': self.y_labeled,
            'x_unlabeled': self.x_unlabeled,
            'y_unlabeled': self.y_unlabeled,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            'x_test': self.x_test,
            'y_test': self.y_test,
        })

    def get_train_size(self):
        train_size = len(self.x_unlabeled)
        if self.args.scenario == 'labels-at-client':
            train_size += len(self.x_labeled)
        return train_size

    def get_task_id(self):
        return self.state['curr_task']

    def get_client_id(self):
        return self.state['client_id']

    def stop(self):
        self.logger.print(self.state['client_id'], 'finished learning all tasks')
        self.logger.print(self.state['client_id'], 'done.')
        self.done = True
