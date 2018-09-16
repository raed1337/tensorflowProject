import nltk
from nltk.ccg import lexicon
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

def create_lexicon(pos,neg):
    lexicon =[]
    lexicon_f =[]
    for fi in [pos,neg]:
        with open(fi ,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            lexicon_f.append(w)
    print(len(lexicon_f))
    return lexicon_f


def sample_handling(sample , lexicon , calassification):
    featureSet  = []

    with open( sample,'r') as f:
        content = f.readline()
        for l in content[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))
            for word in current_words :
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] = 1
            features = list(features)
            featureSet.append([features,calassification])

    return featureSet

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt' , lexicon ,[1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return  train_x ,train_y , test_x , test_y


