""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
import numpy as np
from BPE import *

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    #out = re.sub(r'[^\w\s]', '', text)
    return text.split()

class Features:

    def __init__(self, data_file):
        with open(data_file) as file:
            data = file.read().splitlines()

        # print(len(list(data_split)[0]))

        s = BPE()
        data_split = map(methodcaller("rsplit", "\t", 1), data)
        if len(list(data_split)[0]) != 1:
            print('1')
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts, self.labels = map(list, zip(*data_split))
            self.tokenized_text = [tokenize(text) for text in texts]
            # self.tokenized_text = []
            # for index, text in enumerate(texts):
            #     print(index)
            #     self.tokenized_text.append(s.use(text))
            self.labelset = list(set(self.labels))
        else:
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts = map(list, zip(*data_split))
            self.tokenized_text = [tokenize(text) for text in list(texts)[0]]
            # self.tokenized_text = []
            # for index, text in enumerate(texts):
            #     print(index)
            #     self.tokenized_text.append(s.use(text))

    @classmethod
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features

        V = set()
        word2index = dict()
        index2word = dict()
        for i in range(len(tokenized)):
            for word in tokenized[i]:
                V.add(word)
        for index, word in enumerate(V):
            word2index[word] = index
            index2word[index] = word
        N_words = len(V)
        transformed = np.empty((len(tokenized), N_words))
        for row, sentence in enumerate(tokenized):
            transformed_1 = np.zeros(N_words)
            for word in sentence:
                if word in V:
                    word_idx = word2index[word]
                    transformed_1[word_idx] += 1
            transformed[row] = transformed_1

        label_index = dict()
        for index, label in enumerate(model.labelset):
            label_index[label] = index
        return transformed,word2index,V,label_index

    @classmethod
    def get_features_2(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features

        V = set()
        words_total = list()
        word2index = dict()
        index2word = dict()
        for i in range(len(tokenized)):
            for word in tokenized[i]:
                words_total.append(word)
        m = Counter(words_total)
        sss = sorted(m.items(),key = lambda x:x[1],reverse=True)
        for i in sss[:10000]:
            V.add(i[0])
        for index, word in enumerate(V):
            word2index[word] = index
            index2word[index] = word
        N_words = len(V)
        transformed = np.empty((len(tokenized), N_words))
        for row, sentence in enumerate(tokenized):
            transformed_1 = np.zeros(N_words)
            for word in sentence:
                if word in V:
                    word_idx = word2index[word]
                    transformed_1[word_idx] += 1
            transformed[row] = transformed_1

        label_index = dict()
        for index, label in enumerate(model.labelset):
            label_index[label] = index
        return transformed, word2index, V, label_index

    def get_features_3(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features
        grams = []
        for i in range(len(tokenized)):
            grams.append(cls.generate_ngrams(tokenized[i], 2))
        V = set()
        word2index = dict()
        index2word = dict()
        for i in range(len(grams)):
            for word in grams[i]:
                V.add(tuple(word))
        for index, word in enumerate(V):
            word2index[word] = index
            index2word[index] = word
        N_words = len(V)
        transformed = np.empty((len(grams), N_words))
        for row, sentence in enumerate(grams):
            transformed_1 = np.zeros(N_words)
            for word in sentence:
                if tuple(word) in V:
                    word_idx = word2index[tuple(word)]
                    transformed_1[word_idx] += 1
            transformed[row] = transformed_1

        label_index = dict()
        for index, label in enumerate(model.labelset):
            label_index[label] = index
        return transformed, word2index, V, label_index

    def generate_ngrams(cls,words, WordsToCombine):
        output = []
        for i in range(len(words) - WordsToCombine + 1):
            output.append(words[i:i + WordsToCombine])
        return output

    def get_features_4(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features

        V = set()
        word2index = dict()
        index2word = dict()
        for i in range(len(tokenized)):
            for word in tokenized[i]:
                V.add(word)
        for index, word in enumerate(V):
            word2index[word] = index
            index2word[index] = word
        N_words = len(V)
        transformed = np.empty((len(tokenized), N_words))
        for row, sentence in enumerate(tokenized):
            transformed_1 = np.zeros(N_words)
            for word in sentence:
                if word in V:
                    word_idx = word2index[word]
                    transformed_1[word_idx] = 1
            transformed[row] = transformed_1

        label_index = dict()
        for index, label in enumerate(model.labelset):
            label_index[label] = index
        return transformed,word2index,V,label_index