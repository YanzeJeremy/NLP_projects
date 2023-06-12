import torch.nn as nn
import torch
import preparedata
import data_utils
from itertools import product
import torch.optim as optim
import numpy as np
import embedding
import os
import time

def train():
    vocabulary,pos,labels,number = preparedata.write_train('train.conll')
    model_train = embedding.LinearModel(labels)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    model_train.to(device)
    model_train.form_embedding(vocabulary,pos,labels)
    truth = []
    word_feature = []
    pos_feature = []
    label_feature = []
    with open('train_hhh.converted','r') as file:
        datas = file.readlines()
        for data in datas:
            m = data.strip('\n').split(' ')
            word_feature.append(m[:18])
            pos_feature.append(m[18:36])
            label_feature.append(m[36:48])
            truth.append(m[48:])
    optimizer = optim.Adagrad(model_train.parameters(),lr=0.01,weight_decay=1e-8)
    loss_function = nn.CrossEntropyLoss()
    Feature = data_utils.FeatureGenerator()
    EPOCH = 5
    transitiontensor = torch.zeros(len(truth),1 + 2 * len(labels))
    for i in range(len(transitiontensor)):
        transitiontensor[i][model_train.trainsition2idx[tuple(truth[i])]] = 1
    start_time = time.time()
    for epoch in range(EPOCH):
        print('Epoch:', epoch + 1, 'Training...')
        print("--- %s seconds ---" % (time.time() - start_time))
        for i in range(len(truth)):
            word_vector, pos_vector, label_vector = Feature.form_features(model_train, word_feature[i], pos_feature[i], label_feature[i])
            optimizer.zero_grad()
            pred_out = model_train.forward(word_vector, pos_vector, label_vector)
            target = transitiontensor[i]
            loss = loss_function(pred_out[0], target)
            loss.backward()
            optimizer.step()
        #torch.save(model_train, 'trainhhh-{}.model'.format(epoch))
        print(loss)
    torch.save(model_train,'train_base.model',_use_new_zipfile_serialization=False)


train()



