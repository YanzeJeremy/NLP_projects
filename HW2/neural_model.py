import torch
from model import Model
from nn_layers import FeedForwardNetwork
from operator import methodcaller
import numpy as np
import string
np.random.seed(42)

def tokenize(text):
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    # out = re.sub(r'[^\w\s]', '', text)
    return text.split()

class NeuralModel(Model):
    def __init__(self,u,l,f,b,e,E):
        self.u = u
        self.l = l
        self.f = f
        self.b = b
        self.e = e
        self.E = E
        # might want to save other things here
        pass

    def sentence_split(self,input_file):
        """
        This function used for training file split
        """
        with open(input_file) as file:
            data = file.read().splitlines()
        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, labels = map(list, zip(*data_split))
        tokenized_text = [tokenize(text) for text in texts]
        return tokenized_text,labels


    def sentence_split_classify(self,input_file):
        """
        This function used for testing file split
        """
        with open(input_file) as file:
            data = file.read().splitlines()
        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts = map(list, zip(*data_split))
        tokenized_text = [tokenize(text) for text in list(texts)[0]]
        return tokenized_text

    def feature_select(self,feature_file):
        """
        This function used for word embedding file split
        """
        feature_dict = dict()
        with open(feature_file) as file:
            datas = file.read().splitlines()
        m = [data.split(' ') for data in datas]
        for i in m:
            if '' in i:
                i = i[:-1]
            feature_dict[i[0]] = list(map(float, i[1:]))
        return feature_dict
    
    def train(self, input_file):
        """
        From-scratch Implementation for training model
        """
        sentences, labels = self.sentence_split(input_file)
        labelset = list(set(labels))
        labels1 = [labelset.index(label) for label in labels]
        N_labels = len(labelset)
        list2 = list()
        self.feature_dict = self.feature_select(self.E)
        count = 0
        N_features = len(self.feature_dict['UNK'])*self.f
        for sentence in sentences:
            list1 = list()
            for i in sentence:
                m = i.lower()
                if m not in self.feature_dict.keys():
                    count+=1
                    list1.extend(self.feature_dict['UNK'])
                else:
                    list1.extend(self.feature_dict[m])
            if len(list1) < N_features:
                list1 += [0 for i in range(N_features-len(list1))]
            list2.append(list1[0:N_features])
        input_x = np.array(list2)
        input_y = np.array(labels1)

        self.network = FeedForwardNetwork(self.u,N_features,N_labels)
        for i in range(self.e):
            batches = len(sentences)//self.b #batch size
            print(i)
            for j in range(batches):
                list_batche = input_x[j*self.b:(j+1)*self.b]
                labels_batche = input_y[j*self.b:(j+1)*self.b]
                input,A1,H1,A2,H2 = self.network.forward(list_batche)
                dw1,dw2,db1,db2 = self.network.backward(input,A1,H1,A2,H2,labels_batche)
                self.network.W = self.network.W - self.l * (1.0 / self.b) * dw1
                self.network.B_1 = self.network.B_1 - self.l * (1.0 / self.b) * db1
                self.network.U = self.network.U - self.l * (1.0 / self.b) * dw2
                self.network.B_2 = self.network.B_2 - self.l * (1.0 / self.b) * db2

            input_b, A1_b, H1_b, A2_b, H2_b = self.network.forward(list2)
            loss = self.network.loss(labels1, H2_b)
            argmax = np.argmax(H2_b, axis=1)
            m = 0
            for i in range(len(labels1)):
                if labels1[i] == argmax[i]:
                    m +=1


        self.labelset = labelset
        pass

    def classify(self, input_file):
        """
        From-scratch Implementation for classification
        """
        sentences,labels = self.sentence_split(input_file)
        labels2 = [self.labelset.index(label) for label in labels]
        list2 = list()
        N_features = len(self.feature_dict['UNK']) * self.f
        for sentence in sentences:
            list1 = list()
            for i in sentence:
                m = i.lower()
                if m not in self.feature_dict.keys():
                    list1.extend(self.feature_dict['UNK'])
                else:
                    list1.extend(self.feature_dict[m])
            if len(list1) < N_features:
                list1 += [0 for i in range(N_features - len(list1))]
            list2.append(list1[0:N_features])
        input_x = np.array(list2)
        input_c, A1_c, H1_c, A2_c, H2_c = self.network.forward(input_x)
        # loss = self.network.loss(labels2, H2_c)
        argmax = np.argmax(H2_c, axis=1)
        m = 0
        for i in range(len(labels2)):
            if labels2[i] == argmax[i]:
                m += 1
        print(m/len(labels2))
        pred = []
        for i in argmax:
            pred.append(self.labelset[i])


        return pred

    def train_torch(self,input_file):
        """
        PyTorch Implementation for training model
        """
        sentences, labels = self.sentence_split(input_file)
        labelset = list(set(labels))
        labels1 = [labelset.index(label) for label in labels]
        N_labels = len(labelset)
        list2 = list()
        self.feature_dict = self.feature_select(self.E)
        count = 0
        N_features = len(self.feature_dict['UNK']) * self.f
        for sentence in sentences:
            list1 = list()
            for i in sentence:
                m = i.lower()
                if m not in self.feature_dict.keys():
                    count += 1
                    list1.extend(self.feature_dict['UNK'])
                else:
                    list1.extend(self.feature_dict[m])
            if len(list1) < N_features:
                list1 += [0 for i in range(N_features - len(list1))]
            list2.append(list1[0:N_features])
        input_x = np.array(list2)
        y = np.zeros((len(labels1),N_labels))  # (10,4)
        for index, i in enumerate(labels1):
            y[index][i] = 1
        x_torch = torch.tensor(input_x)
        y_torch = torch.tensor(y)
        #
        self.whole_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=N_features, out_features=self.u),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.u, out_features=N_labels),
            torch.nn.Softmax(dim=1)
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        for t in range(self.e):
            print("epochs" + str(t))
            batches = len(sentences) // self.b
            for j in range(batches):
                batches_x = x_torch[j * self.b:(j + 1) * self.b]
                batches_y = y_torch[j * self.b:(j + 1) * self.b]
                y_pred = self.whole_model(batches_x.float())
                loss = (1.0 / self.b)*loss_fn(y_pred, batches_y)
                self.whole_model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for param in self.whole_model.parameters():
                        param -= self.l * param.grad
            y_pred_2 = self.whole_model(x_torch.float())
            arg1 = torch.argmax(y_pred_2, dim=1)
            arg2 = torch.argmax(y_torch, dim=1)
            m = 0
            for i in range(len(arg2)):
                if arg1[i] == arg2[i]:
                    m +=1
            loss = loss_fn(y_pred_2, y_torch)

        self.labelset = labelset
        pass

    def classify_torch(self,input_file):
        """
        PyTorch Implementation for classification
        """
        sentences,labels = self.sentence_split(input_file)
        labels1 = [self.labelset.index(label) for label in labels]
        N_labels = len(self.labelset)
        list2 = list()
        N_features = len(self.feature_dict['UNK']) * self.f
        for sentence in sentences:
            list1 = list()
            for i in sentence:
                m = i.lower()
                if m not in self.feature_dict.keys():
                    list1.extend(self.feature_dict['UNK'])
                else:
                    list1.extend(self.feature_dict[m])
            if len(list1) < N_features:
                list1 += [0 for i in range(N_features - len(list1))]
            list2.append(list1[0:N_features])
        input_x = np.array(list2)
        y = np.zeros((len(labels1), N_labels))
        for index, i in enumerate(labels1):
            y[index][i] = 1
        x_torch = torch.tensor(input_x)
        y_torch = torch.tensor(y)

        y_pred_2 = self.whole_model(x_torch.float())
        arg1 = torch.argmax(y_pred_2, dim=1)
        arg2 = torch.argmax(y_torch, dim=1)
        m = 0
        for i in range(len(arg2)):
            if arg1[i] == arg2[i]:
                m += 1
        print(m/len(arg2))
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(y_pred_2, y_torch)
        pred = []
        for i in arg1:
            pred.append(self.labelset[i])



        return pred



