import threading
import time
import tensorflow as tf
from scipy import spatial
from scipy.stats import truncnorm
from misc.utils import *
from models.fedmatch.client import Client
from modules.federated import ServerModule
import math

class Server(ServerModule):

    def __init__(self, args):

        super(Server, self).__init__(args, Client)
        self.c2s_sum = []
        self.c2s_sig = []
        self.c2s_psi = []
        self.s2c_sum = []
        self.s2c_sig = []
        self.s2c_psi = []
        self.s2c_hlp = []
        self.restored_clients = {}
        self.rid_to_cid = {}
        self.cid_to_vectors = {}
        self.cid_to_weights = {}
        self.curr_round = -1
        self.nums = 0
        self.weights = 0
        mu, std, lower, upper = 125, 125, 0, 255
        # truncnorm : https://vimsky.com/examples/usage/python-scipy.stats.truncnorm.html
        # rvs：产生服从指定分布的随机数
        self.rgauss = self.loader.scale(truncnorm((lower - mu) / std, (upper - mu) / std,
                                                  loc=mu, scale=std).rvs(
            (1, 32, 32, 3)))  # fixed gaussian noise for model embedding

    def build_network(self):
        self.global_model = self.net.build_resnet9(decomposed=True)

        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        self.trainables = [sig for sig in self.sig]  # only sigma will be updated at server (Labels at Serve scenario)
        self.restored_clients = {i: self.net.build_resnet9(decomposed=False) for i in range(self.args.num_clients)}
        for rid, rm in self.restored_clients.items():
            rm.trainable = False

    def _train_clients(self):
        sigma = [s.numpy() for s in self.sig]
        psi = [p.numpy() for p in self.psi]

        while len(self.connected_ids) > 0:
            for gpu_id, gpu_client in self.clients.items():
                cid = self.connected_ids.pop(0)
                # helpers = self.get_similar_models(cid)
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    # each client will be trained in parallel
                    if self.curr_round == 0:
                        slope = 1
                    else:
                        slope = self.args.num_clients * self.clients_round_weight[cid].get('weight') / self.weights
                    thrd = threading.Thread(target=self.invoke_client,
                                            args=(gpu_client, cid, self.curr_round, sigma, psi, slope))
                    # self.threads.append(thrd)
                    thrd.start()
                if len(self.connected_ids) == 0:
                    break

        # 训练服务器 有监督模型
        thrd_server = threading.Thread(target=self.train_global_model(), args=())
        thrd_server.start()
        thrd_server.join()

        # 等待 大部分线程运行完毕
        while self.queue.qsize() < int(round(self.args.num_clients * self.args.frac_clients)):
            time.sleep(1)


        for x in range(self.queue.qsize()):
            update = self.queue.get()
            self.connected_ids = self.connected_ids.append(update[2])

            tmp = {}
            tmp['last_round'] = update[3]
            client = self.clients_round_weight[update[2]]
            if client == 100:
                tmp['weight'] = math.pow(math.exp() / 2, self.curr_round + 1)
            else:
                tmp['weight'] = client.get('weight') + math.pow(math.exp()/2, self.curr_round+1)

            # 需要加上参数
            self.clients_round_weight.pop(update[2])
            self.clients_round_weight.insert(update[2], tmp)

            self.updates.append(update)
            if self.args.scen == 0:
                self.nums = self.nums + self.basic_nums[update[2]]
            elif self.args.scen == 1:
                self.nums = self.nums + self.balance_nums[update[2]]
            else:
                self.nums = self.nums + self.mix_nums[update[2]]

        self.weights = 0
        for i in range(self.args.num_clients) :
            weight = 0
            if self.clients_round_weight[i] != 100:
                weight = self.clients_round_weight[i].get('weight')
                self.weights += weight
        self.set_weights(self.aggregate(self.updates, self.trainables, self.nums, self.curr_round))

        self.nums = 0

        self.train.evaluate_after_aggr()

        self.logger.save_current_state('server', {
            'c2s': {
                'sum': self.c2s_sum,
                'sig': self.c2s_sig,
                'psi': self.c2s_psi,
            },
            's2c': {
                'sum': self.s2c_sum,
                'sig': self.s2c_sig,
                'psi': self.s2c_psi,
                'hlp': self.s2c_hlp,
            },
            'scores': self.train.get_scores()
        })
        self.updates = []

    def invoke_client(self, client, cid, curr_round, sigma, psi, slope):
        update = client.train_one_round(cid, curr_round, sigma=sigma, psi=psi, slope=slope)
        self.queue.put(update)

    def client_similarity(self, updates):
        self.restore_clients(updates)
        for rid, rmodel in self.restored_clients.items():
            cid = self.rid_to_cid[rid]
            # squeeze :  https://blog.csdn.net/zenghaitao0128/article/details/78512715  从数组的形状中删除单维度条目，即把shape中为1的维度去掉
            self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss))  # embed models
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)

    def restore_clients(self, updates):
        rid = 0
        self.rid_to_cid = {}
        for cwgts, csize, cid, _, _ in updates:
            self.cid_to_weights[cid] = cwgts
            rwgts = self.restored_clients[rid].get_client_train_weights()
            if self.args.scenario == 'labels-at-client':
                half = len(cwgts) // 2
                for lid in range(len(rwgts)):
                    rwgts[lid] = cwgts[lid] + cwgts[lid + half]  # sigma + psi
            elif self.args.scenario == 'labels-at-server':
                for lid in range(len(rwgts)):
                    rwgts[lid] = self.sig[lid].numpy() + cwgts[lid]  # sigma + psi
            self.restored_clients[rid].set_weights(rwgts)
            self.rid_to_cid[rid] = cid
            rid += 1

    def get_similar_models(self, cid):
        if cid in self.cid_to_vectors and (self.curr_round + 1) % self.args.h_interval == 0:
            cout = self.cid_to_vectors[cid]
            sims = self.tree.query(cout, self.args.num_helpers + 1)
            hids = []
            weights = []
            for vid in sims[1]:
                selected_cid = self.vid_to_cid[vid]
                if selected_cid == cid:
                    continue
                w = self.cid_to_weights[selected_cid]
                if self.args.scenario == 'labels-at-client':
                    half = len(w) // 2
                    w = w[half:]
                weights.append(w)
                hids.append(selected_cid)
            return weights[:self.args.num_helpers]
        else:
            return None

    def set_weights(self, new_weights):
        if self.args.scenario == 'labels-at-client':
            half = len(new_weights) // 2
            for i, nwghts in enumerate(new_weights):
                if i < half:
                    self.sig[i].assign(new_weights[i])
                else:
                    self.psi[i - half].assign(new_weights[i])
        elif self.args.scenario == 'labels-at-server':
            for i, nwghts in enumerate(new_weights):
                self.psi[i].assign(new_weights[i])

    def avg_c2s(self):  # client-wise average
        ratio_list = []
        sig_list = []
        psi_list = []
        for upd in self.updates:
            c2s = upd[3]
            ratio_list.append(c2s['ratio'][-1])
            sig_list.append(c2s['sig_ratio'][-1])
            psi_list.append(c2s['psi_ratio'][-1])
        try:
            self.c2s_sum.append(np.mean(ratio_list, axis=0))
            self.c2s_sig.append(np.mean(sig_list, axis=0))
            self.c2s_psi.append(np.mean(psi_list, axis=0))
        except:
            pdb.set_trace()

    def avg_s2c(self):  # client-wise average
        sum_list = []
        sig_list = []
        psi_list = []
        hlp_list = []
        for upd in self.updates:
            s2c = upd[4]
            sum_list.append(s2c['ratio'][-1])
            sig_list.append(s2c['sig_ratio'][-1])
            psi_list.append(s2c['psi_ratio'][-1])
            hlp_list.append(s2c['hlp_ratio'][-1])
        self.s2c_sum.append(np.mean(sum_list, axis=0))
        self.s2c_sig.append(np.mean(sig_list, axis=0))
        self.s2c_psi.append(np.mean(psi_list, axis=0))
        self.s2c_hlp.append(np.mean(hlp_list, axis=0))
