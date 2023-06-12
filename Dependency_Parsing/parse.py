import argparse
import time
import torch
import preparedata
import data_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-m', type=str, help='model file')
    parser.add_argument('-i', type=str, help='input file')
    parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()

    model = torch.load(args.m,map_location=torch.device('cpu'))
    model.eval()
    Feature = data_utils.FeatureGenerator()

    document_dev = preparedata.write_dev(args.i)
    string = ''
    for i in document_dev:
        while i.buffer or len(i.stack) != 1:
            words_feature_dev, pos_feature_dev, label_feature_dev = Feature.extract_features(i)
            words_vector_dev, pos_vector_dev, label_vector_dev = Feature.form_features(model, words_feature_dev,
                                                                                       pos_feature_dev,
                                                                                       label_feature_dev)
            pred_out = model.forward(words_vector_dev, pos_vector_dev, label_vector_dev)
            id2 = torch.topk(pred_out, 2).indices

            check = i.check_trans(model.idx2trainsition[id2[0][0].item()][0])
            if check is True:
                i.update_state_dev(model.idx2trainsition[id2[0][0].item()][0], i,
                                   model.idx2trainsition[id2[0][0].item()][1])
            else:
                i.update_state_dev(model.idx2trainsition[id2[0][1].item()][0], i,
                                   model.idx2trainsition[id2[0][1].item()][1])
        for j in range(1,len(i.tokens)):
            m = i.tokens[j]
            string += str(m.token_id)+'\t'+str(m.word)+'\t'+str(m.word)+'\t'+str(m.pos)\
                      +'\t'+str(m.pos)+'\t'+'_'+'\t'+str(m.predicted_head)+'\t'+\
                      str(m.predicted_dep)+'\t'+'_'+'\t'+'_'+'\n'
        string +='\n'

    with open(args.o, 'w') as file:
        file.write(string)
    #model.eval()


