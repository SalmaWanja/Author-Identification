import sklearn
import flask
import flask_cors
from flask import request, jsonify, Flask
from flask_cors import CORS
import statistics
from math import sqrt
from math import pi
from math import exp

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


def naive_bayes():
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

    # FEATURE STOPWORD FREQUENCY
    class1_stopwords_frequency_arr = []
    class2_stopwords_frequency_arr = []
    class3_stopwords_frequency_arr = []
    # get stopword frequencies, calculate mean and stdeviation
    i = 0
    while i < 9:
        class1_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[i][1]))
        class2_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[9+i][1]))
        class3_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[18+i][1]))
        i += 1

    # class 1
    class1_stopwords_frequency_mean = statistics.mean(
        class1_stopwords_frequency_arr)
    class1_stopwords_frequency_stdev = statistics.stdev(
        class1_stopwords_frequency_arr)
    # class2
    class2_stopwords_frequency_mean = statistics.mean(
        class2_stopwords_frequency_arr)
    class2_stopwords_frequency_stdev = statistics.stdev(
        class2_stopwords_frequency_arr)
    # class3
    class3_stopwords_frequency_mean = statistics.mean(
        class3_stopwords_frequency_arr)
    class3_stopwords_frequency_stdev = statistics.stdev(
        class3_stopwords_frequency_arr)

    # FEATURE SENTENCE DISTRIBUTION
    # SIMPLE SENTENCE DISTRIBUTION
    class1_simple_sentence_distribution_arr = []
    class2_simple_sentence_distribution_arr = []
    class3_simple_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[0])
        class2_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[9+i][1])[0])
        class3_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[18+i][1])[0])
        i += 1

     # class 1
    class1_simple_sentence_distribution_mean = statistics.mean(
        class1_simple_sentence_distribution_arr)
    class1_simple_sentence_distribution_stdev = statistics.stdev(
        class1_simple_sentence_distribution_arr)
    # class2
    class2_simple_sentence_distribution_mean = statistics.mean(
        class2_simple_sentence_distribution_arr)
    class2_simple_sentence_distribution_stdev = statistics.stdev(
        class2_simple_sentence_distribution_arr)
    # class3
    class3_simple_sentence_distribution_mean = statistics.mean(
        class3_simple_sentence_distribution_arr)
    class3_simple_sentence_distribution_stdev = statistics.stdev(
        class3_simple_sentence_distribution_arr)

    # COMPOUND SENTENCE DISTRIBUTION
    class1_compound_sentence_distribution_arr = []
    class2_compound_sentence_distribution_arr = []
    class3_compound_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[1])
        class2_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[9+i][1])[1])
        class3_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[18+i][1])[1])
        i += 1
     # class 1
    class1_compound_sentence_distribution_stdev = statistics.stdev(
        class1_compound_sentence_distribution_arr)
    class1_compound_sentence_distribution_mean = statistics.mean(
        class1_compound_sentence_distribution_arr)
    # class2
    class2_compound_sentence_distribution_mean = statistics.mean(
        class2_compound_sentence_distribution_arr)
    class2_compound_sentence_distribution_stdev = statistics.stdev(
        class2_compound_sentence_distribution_arr)
    # class3
    class3_compound_sentence_distribution_mean = statistics.mean(
        class3_compound_sentence_distribution_arr)
    class3_compound_sentence_distribution_stdev = statistics.stdev(
        class3_compound_sentence_distribution_arr)

    # COMPLEX SENTENCE DISTRIBUTION
    class1_complex_sentence_distribution_arr = []
    class2_complex_sentence_distribution_arr = []
    class3_complex_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[2])
        class2_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[9+i][1])[2])
        class3_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[18+i][1])[2])
        i += 1
     # class 1
    class1_complex_sentence_distribution_stdev = statistics.stdev(
        class1_complex_sentence_distribution_arr)
    class1_complex_sentence_distribution_mean = statistics.mean(
        class1_complex_sentence_distribution_arr)
    # class2
    class2_complex_sentence_distribution_mean = statistics.mean(
        class2_complex_sentence_distribution_arr)
    class2_complex_sentence_distribution_stdev = statistics.stdev(
        class2_complex_sentence_distribution_arr)
    # class3
    class3_complex_sentence_distribution_mean = statistics.mean(
        class3_complex_sentence_distribution_arr)
    class3_complex_sentence_distribution_stdev = statistics.stdev(
        class3_complex_sentence_distribution_arr)

    class1_stats = {'stopwordsFrequency': {'mean': class1_stopwords_frequency_mean, 'stdev': class1_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class1_simple_sentence_distribution_mean, 'stdev': class1_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class1_compound_sentence_distribution_mean, 'stdev': class1_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class1_complex_sentence_distribution_mean, 'stdev': class1_complex_sentence_distribution_stdev}}
    class2_stats = {'stopwordsFrequency': {'mean': class2_stopwords_frequency_mean, 'stdev': class2_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class2_simple_sentence_distribution_mean, 'stdev': class2_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class2_compound_sentence_distribution_mean, 'stdev': class2_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class2_complex_sentence_distribution_mean, 'stdev': class2_complex_sentence_distribution_stdev}}
    class3_stats = {'stopwordsFrequency': {'mean': class3_stopwords_frequency_mean, 'stdev': class3_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class3_simple_sentence_distribution_mean, 'stdev': class3_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class3_compound_sentence_distribution_mean, 'stdev': class3_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class3_complex_sentence_distribution_mean, 'stdev': class3_complex_sentence_distribution_stdev}}

    test_classifier(class1_stats, class2_stats, class3_stats, dataset,
                    class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio)


def calculate_probability(x, mean, stdev):
    var = float(stdev)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def test_classifier(class1_stats, class2_stats, class3_stats, dataset, class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio):
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3}

    i = 0
    while i < 3:
        stopwords_frequency = stopwords_frequency_ratio(dataset[25+i][1])
        simple_sentence_distribution = sentence_length_distribution(
            dataset[25+i][1])[0]
        compound_sentence_distribution = sentence_length_distribution(
            dataset[25+i][1])[1]
        complex_sentence_distribution = sentence_length_distribution(
            dataset[25+i][1])[2]

        # get probabilities and compare classes
        # class1
        probability_stopwords_frequency = calculate_probability(
            stopwords_frequency, class1_stats['stopwordsFrequency']['mean'], class1_stats['stopwordsFrequency']['stdev'])
        probability_simple_sentence_distribution = calculate_probability(
            simple_sentence_distribution, class1_stats['simpleSentenceDistribution']['mean'], class1_stats['simpleSentenceDistribution']['stdev'])
        probability_compound_sentence_distribution = calculate_probability(
            compound_sentence_distribution, class1_stats['compoundSentenceDistribution']['mean'], class1_stats['compoundSentenceDistribution']['stdev'])
        probability_complex_sentence_distribution = calculate_probability(
            complex_sentence_distribution, class1_stats['complexSentenceDistribution']['mean'], class1_stats['complexSentenceDistribution']['stdev'])
        # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
        #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
        class1_probability = (probability_stopwords_frequency*probability_simple_sentence_distribution *
                              probability_compound_sentence_distribution*probability_complex_sentence_distribution)*class1_occurrence_ratio
        # print(class1_probability)
        # class2
        probability_stopwords_frequency = calculate_probability(
            stopwords_frequency, class2_stats['stopwordsFrequency']['mean'], class2_stats['stopwordsFrequency']['stdev'])
        probability_simple_sentence_distribution = calculate_probability(
            simple_sentence_distribution, class2_stats['simpleSentenceDistribution']['mean'], class2_stats['simpleSentenceDistribution']['stdev'])
        probability_compound_sentence_distribution = calculate_probability(
            compound_sentence_distribution, class2_stats['compoundSentenceDistribution']['mean'], class2_stats['compoundSentenceDistribution']['stdev'])
        probability_complex_sentence_distribution = calculate_probability(
            complex_sentence_distribution, class2_stats['complexSentenceDistribution']['mean'], class2_stats['complexSentenceDistribution']['stdev'])
        # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
        #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
        class2_probability = (probability_stopwords_frequency*probability_simple_sentence_distribution *
                              probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class2_occurrence_ratio
        # print(class2_probability)
        # class3
        probability_stopwords_frequency = calculate_probability(
            stopwords_frequency, class3_stats['stopwordsFrequency']['mean'], class3_stats['stopwordsFrequency']['stdev'])
        probability_simple_sentence_distribution = calculate_probability(
            simple_sentence_distribution, class3_stats['simpleSentenceDistribution']['mean'], class3_stats['simpleSentenceDistribution']['stdev'])
        probability_compound_sentence_distribution = calculate_probability(
            compound_sentence_distribution, class3_stats['compoundSentenceDistribution']['mean'], class3_stats['compoundSentenceDistribution']['stdev'])
        probability_complex_sentence_distribution = calculate_probability(
            complex_sentence_distribution, class3_stats['complexSentenceDistribution']['mean'], class3_stats['complexSentenceDistribution']['stdev'])
        # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
        #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
        class3_probability = (probability_stopwords_frequency*probability_simple_sentence_distribution *
                              probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class3_occurrence_ratio
        # print(class3_probability)
        # print(class1_probability, class2_probability, class3_probability)
        if(class1_probability > class2_probability and class1_probability > class3_probability):
            print('Class 1 - Faith Oneya')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Class 2 - Bitange Ndemo')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Class 3 - Abigail Arunga')
        i += 1