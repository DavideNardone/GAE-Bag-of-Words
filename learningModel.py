from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
import pandas as pd
import nltk
import re
from time import time
import random
import cPickle as pickle
import numpy as np

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def classifier(clf, X_train, X_test, train_target, test_target):

    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, train_target)

    train_time = time() - t0
    print("Train time: %0.3fs" % train_time)

    print 'Saving TF-IDF models...'
    pickle.dump(clf, open('classifier.pk', 'wb'))

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(test_target, pred)
    print("Accuracy score: %0.3f" % score)


def getScore(clf,x_test):
    gt = clf.classes_
    probs = clf.predict_proba(x_test)
    ind_sort = np.fliplr(np.argsort(probs))

    return gt[ind_sort]


def singleArea(data,dict):

    class Inline(object):
        pass

    data_out = Inline()

    target = []
    features = []
    for quest in data:
        # print(data)
        if quest[2] in dict:
            target.append(quest[2])
            qs = quest[3:]
            # qs = ' '.join(qs).replace(";", " ")

            qs = ' '.join(str(v) for v in qs)
            qs = qs.replace(";", " ")

            #TOKENING
            # tmp = nltk.word_tokenize(tmp) ## different way
            tokenizer = nltk.RegexpTokenizer(r'\w+')
            qs = tokenizer.tokenize(qs)

            #STOPPING
            stop = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('italian')]
            text = [word for word in qs if word not in stop]

            # adding string to dict
            #TODO: fix unrecognized characters (instead of using re.sub(...))
            text = ' '.join(text).replace(";", " ")
            text = re.sub('[\W_]+', ' ', text, flags=re.LOCALE)

            #converting string to list
            text = text.split(' ')

            #STEMMING
            stemmer = nltk.SnowballStemmer('italian')
            text = [stemmer.stem(word) for word in text]

            #converting list to string
            text = ' '.join(text)

            features.append(text.lower())


    data_out.data = features
    data_out.target = target

    return data_out



def groupingArea(data,dict):

    class Inline(object):
        pass

    data_out = Inline()

    #looping on training_set data
    for quest in data:
        if quest[2] in dict:
            # removing unless data for bag-of-words analysis
            qs = quest[3:]
            # removing nan string values
            # tmp = [x for x in tmp if str(x) != 'nan']

            # converting list to string and replacing semicolumn with space
            #solved problem with integer character
            qs = ' '.join(str(v) for v in qs)
            qs = qs.replace(";", " ")

            #TOKENING
            tokenizer = nltk.RegexpTokenizer(r'\w+')
            qs = tokenizer.tokenize(qs)

            #STOPPING (dealing with stressed characters)
            stop = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('italian')]
            text = [word for word in qs if word not in stop]

            # adding string to dict
            #TODO: fix unrecognized characters (instead of using re.sub(...))
            text = ' '.join(text)
            text = re.sub('[\W_]+', ' ', text, flags=re.LOCALE)

            #converting string to list
            text = text.split(' ')

            #STEMMING
            stemmer = nltk.SnowballStemmer('italian')
            text = [stemmer.stem(word) for word in text]

            #converting list to string
            text = ' '.join(text)

            dict.get(quest[2]).append(text.lower())

    # grouping list of document
    D = []
    y_train = []
    for key in dict:
        y_train.append(key)
        tmp = dict.get(key)
        tmp = ' '.join(tmp)
        D.append(tmp)

    data_out.data = D
    data_out.target =y_train

    return data_out


'''
    IC: Informazioni Cronologiche
    CDL: Corso di Laurea
    QD: Aread Didattica
    Q1,Q2,...,Q7: domande
'''
def learning():

    colnames = ['IC', 'CDL', 'AD', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']


    ########## READING XLSX FORMAT ##########
    data = pd.read_excel(
        open('/data/survey.xlsx', 'rb'),
        names=colnames,
        skiprows=1,
        sheetname='Risposte del modulo 1')
    data.dropna(axis='columns', how='any', inplace=True)


    perc_train = 1
    indices = list(range(data.shape[0]))
    random.shuffle(indices)
    print('training_set:' + str(perc_train * len(indices)))
    print('testing_set:' + str((1 - perc_train) * len(indices)))
    data_train = data.values[:int(len(indices) * perc_train)]
    data_test = data.values[len(data_train):]


    dict = {
        'Sanitaria e Medica': [],
        'Scienze e Tecnologie': [],
        'Umanistica e Giuridica': [],
        'Economica': [],
        'Civile e Architettura': []
    }

    # CREATING TRAINING_SET GROUPED
    data_train_out = groupingArea(data_train, dict)
    data_test_out = singleArea(data_test, dict)


    # SECOND SOLUTION
    print("Extracting features from the training data using a sparse vectorizer...")
    t0 = time()
    vectorizer = TfidfVectorizer(min_df=3)
    X_train = vectorizer.fit_transform(data_train_out.data)
    duration = time() - t0
    print("training done in %fs" % (duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    # feature_names = vectorizer.get_feature_names()

    # saving tf-idf models
    with open('TfidfVectorizer.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)

    print("Extracting features from the test data using the same vectorizer")
    # creating data_testing
    X_test = vectorizer.transform(data_test_out.data)
    print("n_samples: %d, n_features: %d" % X_test.shape)


    classifier(MultinomialNB(alpha=.01),X_train, data_train_out.target, data_test.target)


if __name__ == '__main__':
    learning()
