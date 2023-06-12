import torch.nn as nn
import torch
from itertools import product

class LinearModel(nn.Module):
    def __init__(self,label):
        super(LinearModel, self).__init__()
        self.label = label
        self.linear = nn.Linear(2400, 100)
        self.linear2 = nn.Linear(100, 1+2*len(self.label), bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, x3):
        y_pred = self.linear(torch.cat((x1,x2,x3),1))
        activate = self.tanh(y_pred)
        after_drop = self.dropout(activate)
        output = self.linear2(after_drop)
        return output

    def form_embedding(self,vocabulary,pos,label):
        vocabulary.add('<null>')
        pos.add('<null>')
        label.add('<null>')
        vocabulary.add('<unk>')
        pos.add('<unk>')
        label.add('<unk>')
        self.v_embedding = nn.Embedding(len(vocabulary),50)
        self.pos_embedding = nn.Embedding(len(pos), 50)
        self.label_embedding = nn.Embedding(len(label), 50)
        self.word2idx = {word: ind for ind, word in enumerate(vocabulary)}
        self.pos2idx = {pos: ind for ind, pos in enumerate(pos)}
        self.label2idx = {label: ind for ind, label in enumerate(label)}

        label.remove('<null>')
        label.remove('<unk>')
        list_t = ['right_arc', 'left_arc']
        self.trainsition_list = list(product(list_t, label))
        self.trainsition_list.append(tuple(['shift', 'null']))
        self.trainsition2idx = {word: ind for ind, word in enumerate(self.trainsition_list)}
        self.idx2trainsition = {ind: word for ind, word in enumerate(self.trainsition_list)}

