import embedding
import data_utils


def read_conll(file):
    with open(file) as f:
        tokens,arc,document,vocabulary,pos,label,number = [],[],[],set(),set(),set(),0
        for line in f.readlines():
            conll = line.strip().split('\t')
            if len(conll) == 10:
                vocabulary.add(conll[1])
                pos.add(conll[3])
                label.add(conll[7])
                arc.append((int(conll[6]),int(conll[0])))
                token = data_utils.Token(int(conll[0]),conll[1],conll[3],int(conll[6]),conll[7])
                tokens.append(token)
            else:
                sentence = data_utils.Sentence(tokens,arc)
                if sentence.is_projective() == True:
                    document.append(sentence)
                else:
                    number +=1
                tokens, arc = [], []
    return document,vocabulary,pos,label,number

def read_conll_dev(file):
    with open(file) as f:
        tokens,arc,document,vocabulary,pos = [],[],[],set(),set()
        for line in f.readlines():
            conll = line.strip().split('\t')
            if len(conll) == 10:
                vocabulary.add(conll[1])
                pos.add(conll[3])
                token = data_utils.Token(int(conll[0]),conll[1],conll[3],head=conll[2],dep ='<null>')
                tokens.append(token)
            else:
                sentence = data_utils.Sentence(tokens,arc)
                document.append(sentence)
                tokens, arc = [], []
    return document


def write_dev(file):
    document = read_conll_dev(file)
    return document

def write_train(file):
    document,vocabulary,pos,label,number = read_conll(file)
    print(len(document))
    Feature = data_utils.FeatureGenerator()
    string = ''
    count = 0
    for i in document:
        count += 1
        while i.buffer or i.stack != [0]:
            words_feature, pos_feature, label_feature = Feature.extract_features(i)
            transition = i.get_trans()
            pred, truth = i.update_state(transition, i)
            temp = words_feature + pos_feature + label_feature+list(truth)
            string += ' '.join(map(str, temp))+'\n'

    with open('train_hhh.converted','w') as file:
       file.write((string))

    return vocabulary,pos,label,number
