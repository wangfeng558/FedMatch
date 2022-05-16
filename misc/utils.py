import os
import pdb
import json
import torch
import random
import numpy as np
from datetime import datetime

def np_save(base_dir, filename, data):
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)

def np_load(base_dir, filename):
    #load方法读取Numpy专用的二进制数据文件   allow_pickle：布尔型。决定是否加载存储在npy文件的pickled对象数组，默认为True。
    #Python 字典 (Dictionary) items () 函数以列表返回可遍历的 (键, 值) 元组数组。
    return np.load(os.path.join(base_dir, filename), allow_pickle=True).item()

def write_file(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)

def debugger():
    pdb.set_trace()

