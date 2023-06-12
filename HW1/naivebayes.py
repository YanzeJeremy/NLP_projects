"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""
import itertools

from Model import *
from Features import *
from collections import Counter , defaultdict
import numpy as np
class NaiveBayes(Model):
    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        document_dict = defaultdict(lambda:list())
        logprior = dict()
        bigdoc = defaultdict(lambda:list())
        loglikelihood = dict()
        features = Features(input_file)
        output,word2index,V,label_index = features.get_features_2(features.tokenized_text,features)
        print(output.shape)
        print(len(V))
        for row,sentence in enumerate(output):
            bigdoc[features.labels[row]].append(sentence)
            document_dict[features.labels[row]].append(sentence)
        for key in document_dict.keys():
            print(key)
            N_c = len(document_dict[key])
            logprior[key] = np.log(N_c / len(output))
            count_key = bigdoc[key][0]

            for i in range(1,len(bigdoc[key])):
                count_key += bigdoc[key][i]
            for w in V:
                count_wc = count_key[word2index[w]]
                loglikelihood[w, key] = np.log((count_wc + 1) / (sum(count_key) + len(V)))

        # x = 3
        # breakpoint()
        model = logprior,loglikelihood,V,features.labelset
        #model = None
        #model = self.fit(input_file)
        # Save the model
        self.save_model(model)
        return model

    def fit(self,input_file):
        document_dict = defaultdict(lambda:list())
        logprior = dict()
        bigdoc = defaultdict(lambda:list())
        loglikelihood = dict()
        N_doc = 0
        V = set()

        features =Features(input_file)
        for row,sentence in enumerate(features.tokenized_text):
            N_doc += 1
            for word in sentence:
                V.add(word)
                bigdoc[features.labels[row]].append(word)
            document_dict[features.labels[row]].append(sentence)

        for key in document_dict.keys():
            N_c = len(document_dict[key])
            logprior[key] = np.log(N_c / N_doc)
            count_key = Counter(bigdoc[key])
            for w in V:
                count_wc = count_key[w]
                loglikelihood[w, key] = np.log((count_wc + 1) / (sum(count_key.values()) + len(V)))

        return logprior,loglikelihood,V,features.labelset





    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 
        ## TODO write your code here
        # preds = []
        # sum = dict()
        # logprior,loglikelihood,V,labels = model
        #
        #
        # file = Features(input_file)
        # labels_answer = file.labels
        #
        # for test2 in file.tokenized_text:
        #     for label in labels:
        #         sum[label] = logprior[label]
        #         for i in range(len(test2)):
        #             word = test2[i]
        #             if word in V:
        #                 sum[label] = sum[label]+loglikelihood[word,label]
        #     sorted_dict = sorted(sum.items(), key=lambda x: x[1],reverse = True)
        #     preds.append(sorted_dict[0][0])
        # count = 0
        # for i in range(len(preds)):
        #     if preds[i] == labels_answer[i]:
        #         count +=1
        # acc = count/len(preds)
        # print(acc)

        preds = []
        sum = dict()
        logprior, loglikelihood, V, labels = model

        file = Features(input_file)
        grams = []
        for i in range(len(file.tokenized_text)):
            grams.append(file.generate_ngrams(file.tokenized_text[i], 2))
        labels_answer = file.labels

        for test2 in grams:
            for label in labels:
                sum[label] = logprior[label]
                for i in range(len(test2)):
                    word = test2[i]
                    if tuple(word) in V:
                        sum[label] = sum[label] + loglikelihood[tuple(word), label]
            sorted_dict = sorted(sum.items(), key=lambda x: x[1], reverse=True)
            preds.append(sorted_dict[0][0])
        count = 0
        for i in range(len(preds)):
            if preds[i] == labels_answer[i]:
                count += 1
        acc = count / len(preds)
        print(acc)
        return preds


