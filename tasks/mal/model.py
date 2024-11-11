import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datetime import datetime
import random
import sklearn.metrics as metrics
import pickle
import json

def load_pkl(p):
    with open(p,'rb') as pkl_file:
        f_out = pickle.load(pkl_file)
    return f_out

def get_dict(path):
    """read embeddings as dictionary
    """
    dic = dict()
    with open(path) as f:
        vec = f.readlines()
        for line in vec[1:]:
            items = line.split()
            dic[items[0]] = [float(item) for item in items[1:]]
    print("{} unique instructions in total. ".format(len(dic)))
    return dic

# read the vectors into a dictionary
target = 'x86'

path_x86_vec = './x86.emb'
dict_x86 = get_dict(path_x86_vec)
path_target_vec = f'./{target}.emb'
dict_target = get_dict(path_target_vec)

x86_benign = './train_benign.pkl' 
x86_malware = './x86_mal.pkl' 
target_benign = f'./{target}_benign.pkl'
target_malware = f'./{target}_mal.pkl'
# read benign dataset in x86
x86_dataset_benign = load_pkl(x86_benign)
# read malware dataset in x86
x86_dataset_malware = load_pkl(x86_malware)


EMB_DIM = 200
MAX_LEN = 10000

random.shuffle(x86_dataset_benign)
random.shuffle(x86_dataset_malware)

# build the dataset
training_size = 800
testing_size = 200

x86_training_set = x86_dataset_benign[:training_size] + x86_dataset_malware[:training_size]
x86_training_labels = [0] * training_size + [1] * training_size


X_train, X_test, y_train, y_test = train_test_split(x86_training_set, x86_training_labels, test_size=0.2, random_state=42)