import logging

from flask import *
import numpy as np
import nltk
import re
import cPickle as pickle
import json
from sklearn.naive_bayes import MultinomialNB

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

app = Flask(__name__)

def getScore(clf,x_test):

    score = {}

    gt = clf.classes_
    probs = clf.predict_proba(x_test)
    ind_sort = np.fliplr(np.argsort(probs))

    score['cls'] = gt[ind_sort].tolist()
    score['probs'] = probs[0][ind_sort].tolist()

    return score

def classifier(clf,features):

    clf = pickle.load(open('/models/classifier.pk', 'rb'))
    res = getScore(clf,features)

    return res


@app.route('/', methods=['GET', 'POST'])
def resize():

    qs = ''
    if request.method == 'POST':
        content = request.get_json(silent=True)
        #handling json request
        for k, v in content.iteritems():
            # handling array
            if type(v) is list:
                for val in v:
                    qs += val + ' '
            else:
                # handling string
                qs += v + ' '
    else:
        resp = Response(response='405',
                        status=405,
                        mimetype="application/json"
                        )
        return (resp)

    # LOADING TF-IDF STRUCTURE
    vectorizer = pickle.load(open('/models/TfidfVectorizer.pk','rb'))

    # DOWNLOADING stopwords
    nltk.download('stopwords')

    # TOKENING
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    qs = tokenizer.tokenize(qs)

    # STOPPING (working)
    stop = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('italian')]
    text = [word for word in qs if word not in stop]

    # STEMMING
    stemmer = nltk.SnowballStemmer('italian')
    text = [stemmer.stem(word) for word in text]

    # converting list to string
    text = ' '.join(text)

    tl = []
    tl.append(text)
    # RETRIEVING FEATURES
    X_test = vectorizer.transform(tl)

    res = classifier(MultinomialNB(alpha=.01),X_test)

    return Response(response = json.dumps({'classes': res}, sort_keys=False, indent=4, encoding='utf-8'),
                    status = 200,
                    mimetype = "application/json"
                    )


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
