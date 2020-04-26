import sklearn
import flask
import flask_cors
from flask import request, jsonify, Flask
from flask_cors import CORS

import nltk
nltk.download('stopwords')
app = Flask(__name__)
CORS(app)


@app.route('/author_id/api/remove_stopwords', methods=['POST'])
def remove_stopwords():
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    req_data = request.get_json()
    raw_text = req_data['raw_text']
    processed_text = ' '.join(
        [word for word in raw_text.split()if word not in english_stop_words])
    return jsonify({'processed_text': processed_text}), 200


@app.route('/author_id/api/stem', methods=['POST'])
def stemmer():
    from nltk.stem import PorterStemmer
    porter = PorterStemmer()
    req_data = request.get_json()
    raw_text = req_data['raw_text']
    processed_text = ' '.join(porter.stem(word)for word in raw_text.split())
    return jsonify({'processed_text': processed_text}), 200

# gets the distribution of words from around the text to figure out how they are used


def getFrequencyDistribution(text):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    vocabulary = set(tokens)
    print(len(vocabulary))
    frequency_dist = nltk.FreqDist(tokens)
    return sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50]

# vercorize text (feature extraction)


def vectorize(xtrain, xtest):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(xtrain)
    test_vectors = vectorizer.transform(xtest)
    print(train_vectors.shape, test_vectors.shape)

# calculating frequency of stop words


def stopwords_frequency(text):
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    stopword_count = 0
    for word in text.split():
        if word in english_stop_words:
            stopword_count = stopword_count + 1
    return stopword_count

# calculating the frequency ratio of stopwords vs text length


def stopwords_frequency_ratio(text):
    stopwords_count = stopwords_frequency(text)
    text_length = len(text.split())
    frequency_ratio = stopwords_count/float(text_length)
    return frequency_ratio

# calculating sentense length distribution


def sentence_length_distribution(text):
    # breakdown text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    # determine the distribution of sentences
    # simple sentence - has no conjunctions
    # compound sentence = has one conjunction joining two sentences
    # complex sentence = has one or more conjunctions
    # conjunctions can be either coordinating/subordinating
    # coordinating_conjunctions = ["for", "and", "nor", "but", "or", "yet", "so"]
    # subordinating_conjunctions = ["after", "as", "although", "because", "before", "even though", "if", "once", "rather than", "since", "that", "though", "unless", "until", "when", "whenever", "whereas", "while"]
    conjunctions = ["for", "and", "nor", "but", "or", "yet", "so", "after", "as", "although", "because", "before", "even though",
                    "if", "once", "rather than", "since", "that", "though", "unless", "until", "when", "whenever", "whereas", "while"]
    # number of conjunctions in sentences
    conjunction_distribution = []
    for sentence in sentences:
        sentence_conjunction_count = 0
        for word in sentence.split():
            if word in conjunctions:
                sentence_conjunction_count += 1
        conjunction_distribution.append(sentence_conjunction_count)

    simple_sentence_count = 0
    compound_sentence_count = 0
    complex_sentence_count = 0

    # categorizing sentences based on conjunction count
    for count in conjunction_distribution:
        if count == 0:
            simple_sentence_count += 1
        elif count == 1:
            compound_sentence_count += 1
        elif count > 1:
            complex_sentence_count += 1

    simple_sentence_distribution = simple_sentence_count/float(len(sentences))
    compound_sentence_distribution = compound_sentence_count / \
        float(len(sentences))
    complex_sentence_distribution = complex_sentence_count / \
        float(len(sentences))

    return [simple_sentence_distribution, compound_sentence_distribution, complex_sentence_distribution]


def load_dataset():
    # read file
    import csv
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3}
    dataset = []
    with open('articles.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                dataset.append([labels[row[0]], row[1]])
                line_count += 1
    return dataset

# naive bayes - assumes that prodictors/features used are independent of one another
# out features include:
    # frequency of stopwords as a ratio
    # sentence length distribution
    # simple sentences
    # compound sentences
    # complex sentences


def multinomial_naive_bayes():
    dataset = load_dataset()
    # dataset size
    dataset_size = len(dataset)
    # count occurrences of each class
    class1_counter = 0
    class2_counter = 0
    class3_counter = 0
    for data_item in dataset:
        if data_item[0] == 1:
            class1_counter += 1
        elif data_item[0] == 2:
            class2_counter += 1 
        elif data_item[0] == 3:
            class3_counter += 1
    class1_occurrence_ratio = class1_counter/float(dataset_size)
    class2_occurrence_ratio = class2_counter/float(dataset_size) 
    class3_occurrence_ratio = class3_counter/float(dataset_size) 
    
    print(class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio)
