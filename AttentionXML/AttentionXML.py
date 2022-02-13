import os
import re
import click
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import joblib
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.datasets import load_svmlight_file
from gensim.models import KeyedVectors
from tqdm import tqdm
from typing import Union, Iterable

from deepxml.data_utils import *
from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model
from deepxml.tree import FastAttentionXML
from deepxml.networks import AttentionRNN

# model相关配置
model_hidden_size = 256 # 隐藏层大小
model_layers_num = 1 # 层数
model_linear_size = [256] 
model_dropout = 0.5
model_emb_trainable = False
model_train_batch_size = 40 # 训练集 batch_size 大小
model_valid_batch_size = 40 # 验证集 batch_size 大小
model_train_nb_epoch = 30   # 训练集 epoch 大小
model_train_swa_warmup = 4  # 

# data相关配置
data_emb_init = np.load('./data/embedding_matrix.npy') # embedding matrix shape:(所有词的个数，词向量大小)
data_emb_size = 300  # embedding matrix size
data_labels_binarizer_path = './data/label_binarizer' # labels binarizer
data_train_texts = np.load('./data/train_texts_tokenizer.npy') # train_texts.npy  训练集的 tokenizer shape:(训练集数目，每条数据的维度)
data_train_labels = np.load('./data/train_labels.npy', allow_pickle=True) # train_labels.npy  训练集的 labels  shape:(训练集数目，[标签]) 每条数据的标签数目是不同的
data_valid_size = 200 # 验证集的大小
data_vocab = np.load('./data/vocab.npy')  # vocab 词集

max_len = 30   # 每个动态的最大长度

model_name = 'AttentionXML'
data_name = 'OnlyTextInformation6683'
model_path = 'modelss'


# --------------------------------------------- Training ---------------------------------------------------

model_path = os.path.join(model_path, F'{model_name}-{data_name}')

train_x, train_labels = data_train_texts, data_train_labels
train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                test_size=data_valid_size,
                                                                random_state=42)
mlb = get_mlb(data_labels_binarizer_path, np.hstack((train_labels, valid_labels)))
train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
labels_num = len(mlb.classes_)
print('Number of Labels: {}'.format(labels_num))
print('Size of Training Set: {}'.format(len(train_x)))
print('Size of Validation Set: {}'.format(len(valid_x)))

print('Training')

train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                          model_train_batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False),
                          model_valid_batch_size, num_workers=4)
model = Model(network=AttentionRNN, labels_num=labels_num, model_path=model_path, emb_init=data_emb_init,
              emb_size=data_emb_size,
              hidden_size=model_hidden_size, layers_num=model_layers_num, linear_size=model_linear_size, dropout=model_dropout, emb_trainable=model_emb_trainable)
model.train(train_loader, valid_loader, batch_size=model_train_batch_size, nb_epoch=model_train_nb_epoch, swa_warmup=model_train_swa_warmup)

print('Finish Training')