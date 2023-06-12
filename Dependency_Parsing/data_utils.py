import numpy as np
from collections import defaultdict

import torch

P_PREFIX = '<p>'
L_PREFIX = '<l>'
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'


class Token:

    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.head = head
        self.dep = dep
        self.predicted_head = -1
        self.predicted_dep = NULL
        self.lc, self.rc = [], []

    def reset_states(self):
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []


ROOT_TOKEN = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
NULL_TOKEN = Token(token_id=-1, word=NULL, pos=NULL, head=-1, dep=NULL)
UNK_TOKEN = Token(token_id=-1, word=UNK, pos=UNK, head=-1, dep=UNK)


class Sentence:

    def __init__(self, tokens,arcs):
        self.root = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
        temp = list()
        temp.append(self.root)
        self.tokens = temp+tokens
        self.stack = [self.root.token_id]
        self.buffer = [i.token_id for i in tokens]
        self.arcs = arcs
        self.predicted_arcs = None
        self.arcs_head = defaultdict(list)
        self.id2word = {}
        for i in self.tokens:
            self.id2word[i.token_id] = i.word
        self.id2word['<null>'] = '<null>'
        for j in self.arcs:
            self.arcs_head[j[0]].append(j[1])
        self.stacks = []

    def is_projective(self):
        """ determines if sentence is projective when ground truth given """
        length = len(self.arcs)
        for i in range(0,length):
            for j in range(0,length):
                if min(self.arcs[i]) < min(self.arcs[j]) < max(self.arcs[i]) < max(self.arcs[j]):
                    return False
        return True

    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation from [shift, left_arc, or right_arc] """
        if len(self.stack) <2:
            return 'shift'
        if (self.stack[-1],self.stack[-2]) in self.arcs:
            return 'left_arc'
        elif (self.stack[-2],self.stack[-1]) in self.arcs \
                and self.arcs_head[self.stack[-1]] ==[]:
            return 'right_arc'
        else:
            return 'shift'

    def check_trans(self, potential_trans):
        """ checks if transition can legally be performed"""
        if potential_trans == 'shift':
            if len(self.buffer) >= 1:
                return True
            else:
                return False
        else:
            if len(self.stack) >= 2:
                return True
            else:
                return False

    def update_state(self, curr_trans,sentence, predicted_dep=None):
        """ updates the sentence according to the given transition (may or may not assume legality, you implement) """
        if curr_trans == 'shift':
            self.stack.append(self.buffer[0])
            self.buffer = self.buffer[1:]
            return predicted_dep,tuple([curr_trans,'null'])
        elif curr_trans == 'right_arc':
            predicted_dep = (self.stack[-2], self.stack[-1])

            sentence.tokens[self.stack[-2]].rc.append(self.stack[-1])
            toke = sentence.tokens[self.stack[-1]]
            toke.predicted_dep = toke.dep
            self.arcs_head[self.stack[-2]].remove(self.stack[-1])
            self.stack.remove(self.stack[-1])
            return predicted_dep,tuple([curr_trans,toke.dep])
        elif curr_trans == 'left_arc':
            predicted_dep = (self.stack[-1],self.stack[-2])
            sentence.tokens[self.stack[-1]].lc.append(self.stack[-2])
            toke = sentence.tokens[self.stack[-2]]
            toke.predicted_dep = toke.dep
            self.arcs_head[self.stack[-1]].remove(self.stack[-2])
            self.stack.remove(self.stack[-2])

            return predicted_dep,tuple([curr_trans,toke.dep])

    def update_state_dev(self, curr_trans,sentence, predicted_dep=None):
        """ updates the sentence according to the given transition (may or may not assume legality, you implement) """
        if curr_trans == 'shift':
            self.stack.append(self.buffer[0])
            self.buffer = self.buffer[1:]
        elif curr_trans == 'right_arc':
            sentence.tokens[self.stack[-2]].rc.append(self.stack[-1])
            toke = sentence.tokens[self.stack[-1]]
            toke.predicted_dep = predicted_dep
            toke.predicted_head = self.stack[-2]
            self.stack.remove(self.stack[-1])
        elif curr_trans == 'left_arc':
            sentence.tokens[self.stack[-1]].lc.append(self.stack[-2])
            toke = sentence.tokens[self.stack[-2]]
            toke.predicted_dep = predicted_dep
            toke.predicted_head = self.stack[-1]
            self.stack.remove(self.stack[-2])







class FeatureGenerator:

    def __init__(self):
        pass

    def extract_features(self, sentence):
        """ returns the features for a sentence parse configuration """
        word_features = []
        pos_features = []
        dep_features = []
        #input = torch.LongTensor([self.model.word2idx['book'],self.model.word2idx['the']])
        depth_stack = len(sentence.stack)
        w1, w2, w3 = self.stack_feature_f3(depth_stack, sentence.stack)
        depth_buffer = len(sentence.buffer)
        w4, w5, w6 = self.buffer_feature(depth_buffer, sentence.buffer)

        w7, w8, w9, w10 = self.children_feature(w1,sentence)
        w11, w12, w13, w14 = self.children_feature(w2, sentence)

        w15,w16 = self.children_secondlevel(w1, sentence)
        w17,w18 = self.children_secondlevel(w2, sentence)


        word_features.extend([w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,w11, w12, w13, w14,w15,w16,w17,w18])
        word_features_out = []
        for i in word_features:
            if i == NULL:
                pos_features.append(NULL)
                word_features_out.append(NULL)
            else:
                pos_features.append(sentence.tokens[i].pos)
                word_features_out.append(sentence.tokens[i].word)

        for i in [w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18]:
            if i == NULL:
                dep_features.append(NULL)
            else:
                dep_features.append(sentence.tokens[i].predicted_dep)

        return word_features_out, pos_features, dep_features

    def form_features(self,model,words_feature,pos_feature,label_feature):
        word_indexes = list()
        for w in words_feature:
            if w in model.word2idx.keys():
                word_indexes.append(model.word2idx[w])
            else:
                word_indexes.append(model.word2idx['<unk>'])
        word_vector = model.v_embedding(torch.LongTensor(word_indexes))
        pos_indexes = list()
        for w in pos_feature:
            if w in model.pos2idx.keys():
                pos_indexes.append(model.pos2idx[w])
            else:
                pos_indexes.append(model.pos2idx['<unk>'])

        # pos_indexes = [model.pos2idx[w] for w in pos_feature]
        pos_vector = model.pos_embedding(torch.LongTensor(pos_indexes))
        label_indexes = list()
        for w in label_feature:
            if w in model.label2idx.keys():
                label_indexes.append(model.label2idx[w])
            else:
                label_indexes.append(model.label2idx['<unk>'])

        # label_indexes = [model.label2idx[w] for w in label_feature]
        label_vector = model.label_embedding(torch.LongTensor(label_indexes))
        word_vector = word_vector.reshape(1, 900)
        pos_vector = pos_vector.reshape(1, 900)
        label_vector = label_vector.reshape(1, 600)
        return word_vector,pos_vector,label_vector









    def stack_feature_f3(self, depth, stack):
        if depth >= 4:
            return stack[-1], stack[-2], stack[-3]
        elif depth >= 3:
            return stack[-1], stack[-2], NULL
        elif depth == 2:
            return stack[-1], NULL, NULL
        else:
            return NULL, NULL, NULL


    def buffer_feature(self, depth, buffer):
        if depth >= 3:
            return buffer[0], buffer[1], buffer[2]
        elif depth >= 2:
            return buffer[0], buffer[1], NULL
        elif depth >= 1:
            return buffer[0], NULL,NULL
        else:
            return NULL,NULL,NULL

    def children_feature(self,word,sentence):
        if word == NULL:
            return NULL,NULL,NULL,NULL
        else:
            right = sentence.tokens[word].rc
            left = sentence.tokens[word].lc
            len_r = len(right)
            len_l = len(left)
            if len_r >= 2:
                first_r, second_r = right[-1], right[-2]
            elif len_r >= 1:
                first_r, second_r = right[-1], NULL
            else:
                first_r, second_r = NULL,NULL

            if len_l >= 2:
                first_l, second_l = left[0], left[1]
            elif len_l >= 1:
                first_l, second_l = left[0], NULL
            else:
                first_l, second_l = NULL,NULL
            return first_l,second_l,first_r,second_r

    def children_secondlevel(self,word,sentence):
        if word == NULL:
            return NULL,NULL
        else:
            right = sentence.tokens[word].rc
            left = sentence.tokens[word].lc
            len_r = len(right)
            len_l = len(left)
            if len_r >=1:
                rightmost = right[-1]
                right_right = sentence.tokens[rightmost].rc
                if len(right_right) >= 1:
                    right = right_right[-1]
                else:
                    right = NULL
            else:
                right = NULL

            if len_l >=1:
                leftmost = left[0]
                left_left = sentence.tokens[leftmost].lc
                if len(left_left) >= 1:
                    left = left_left[0]
                else:
                    left = NULL
            else:
                left = NULL

            return left,right

