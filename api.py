import flask
import flask_cors
from flask import request, jsonify, Flask
from flask_cors import CORS

import nltk
nltk.download('stopwords') 
app = Flask(__name__)
CORS(app)

import sklearn


@app.route('/author_id/api/remove_stopwords', methods=['POST'])
def remove_stopwords():     
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    req_data = request.get_json()
    raw_text = req_data['raw_text']
    processed_text = ' '.join([word for word in raw_text.split()if word not in english_stop_words])
    return jsonify({'processed_text':processed_text}),200

@app.route('/author_id/api/stem', methods=['POST'])
def stemmer():
    from nltk.stem import PorterStemmer
    porter = PorterStemmer()
    req_data = request.get_json()
    raw_text = req_data['raw_text']
    processed_text = ' '.join(porter.stem(word)for word in raw_text.split())
    return jsonify({'processed_text':processed_text}),200

# gets the distribution of words from around the text to figure out how they are used
def getFrequencyDistribution(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    vocabulary = set(tokens)
    print(len(vocabulary))
    frequency_dist = nltk.FreqDist(tokens)
    return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]

# vercorize text (feature extraction)
def vectorize(xtrain,xtest):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(xtrain)
    test_vectors = vectorizer.transform(xtest)
    print(train_vectors.shape, test_vectors.shape)

# classifier
