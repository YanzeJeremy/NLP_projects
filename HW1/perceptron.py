"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
from Model import *
from Features import *
import numpy as np

class Perceptron(Model):
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        features = Features(input_file)
        output,word2index,V,label_index = features.get_features_2(features.tokenized_text,features) #bug of words features, which returns a N*len(V) matrix
        print(len(V))
        theta = np.zeros((len(features.labelset),len(V)))
        iterations = 100
        for i in range(iterations):
            print(i)
            for j in range(len(features.labels)):
                multi = np.dot(theta,output[j])
                y_hat = list(multi).index(sorted(multi)[-1])
                y = label_index[features.labels[j]]
                if y !=y_hat:
                    theta[y] = theta[y]+output[j]
                    theta[y_hat] = theta[y_hat]-output[j]
                else:
                    continue


        model = theta,label_index,V,word2index



        ## Save the model
        self.save_model(model)
        return model





    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here (and change return)
        theta,label_index,V,word2index = model
        tests = Features(input_file)
        labels_answer = tests.labels
        index2label = {v: k for k, v in label_index.items()}
        print(theta.shape)
        transformed = np.empty((len(tests.tokenized_text), len(V)))
        print(transformed.shape)
        count = 0
        for row, sentence in enumerate(tests.tokenized_text):
            count +=1
            transformed_1 = np.zeros(len(V))
            for word in sentence:
                if word in V:
                    word_idx = word2index[word]
                    transformed_1[word_idx] += 1
            transformed[row] = transformed_1
        # print(transformed)
        preds = list()
        for i in range(len(transformed)):
            multi = np.dot(theta, transformed[i])
            #print(multi)
            y_hat = list(multi).index(sorted(multi)[-1])
            preds.append(index2label[y_hat])
        print(preds)
        print(label_index)
        count = 0
        for i in range(len(preds)):
            if preds[i] == labels_answer[i]:
                count += 1
        acc = count / len(preds)
        print(acc)





        return preds
