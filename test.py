# -*- coding: utf-8 -*-
import nltk
# import statistics
import scipy.stats
import math
from math import sqrt
from math import pi
from math import exp

import sklearn
import flask
import flask_cors
from flask import request, jsonify, Flask
from flask_cors import CORS


# nltk.download('stopwords')
app = Flask(__name__)
CORS(app)
# nltk.download('stopwords')

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


def average_sentence_length(text):
    sentences = nltk.tokenize.sent_tokenize(text.strip())
    sentence_count = 0
    sentence_lengths = []
    for sentence in sentences:
        sentence_count += 1
        sentence_lengths.append(len(sentence))
    # print(sentence_count, sentence_lengths)
    # calculate average sentence length
    total_length_of_sentences = 0
    for i in sentence_lengths:
        total_length_of_sentences += i
    # print(total_length_of_sentences/sentence_count)
    return(total_length_of_sentences/sentence_count)


# calculating sentense length distribution


def sentence_length_distribution(text):
    # breakdown text into sentences
    sentences = nltk.tokenize.sent_tokenize(text.strip())
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

    simple_sentence_distribution = (
        simple_sentence_count/float(len(sentences)))
    compound_sentence_distribution = (
        compound_sentence_count/float(len(sentences)))
    complex_sentence_distribution = (
        complex_sentence_count/float(len(sentences)))

    return [simple_sentence_distribution, compound_sentence_distribution, complex_sentence_distribution]


def load_dataset(fileName):
    # read file
    import csv
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3, 'Walter Menya': 4, 'Ruth Mbula': 5,
              'David Mwere': 6, 'Justus Ochieng': 7, 'Sam Kiplagat': 8, 'Samuel Owino': 9, 'Joseph Mboya': 10}
    dataset = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                dataset.append([labels[row[0].lstrip()], row[1]])
                line_count += 1
    return dataset


def naive_bayes():
    dataset = load_dataset('dataset.csv')
    # dataset size
    dataset_size = len(dataset)
    # count occurrences of each class
    class1_counter = 0
    class2_counter = 0
    class3_counter = 0
    class4_counter = 0
    class5_counter = 0
    class6_counter = 0
    class7_counter = 0
    class8_counter = 0
    class9_counter = 0
    class10_counter = 0
    for data_item in dataset:
        if data_item[0] == 1:
            class1_counter += 1
        elif data_item[0] == 2:
            class2_counter += 1
        elif data_item[0] == 3:
            class3_counter += 1
        elif data_item[0] == 4:
            class4_counter += 1
        elif data_item[0] == 5:
            class5_counter += 1
        elif data_item[0] == 6:
            class6_counter += 1
        elif data_item[0] == 7:
            class7_counter += 1
        elif data_item[0] == 8:
            class8_counter += 1
        elif data_item[0] == 9:
            class9_counter += 1
        elif data_item[0] == 10:
            class10_counter += 1
    class1_occurrence_ratio = class1_counter/float(dataset_size)
    class2_occurrence_ratio = class2_counter/float(dataset_size)
    class3_occurrence_ratio = class3_counter/float(dataset_size)
    class4_occurrence_ratio = class4_counter/float(dataset_size)
    class5_occurrence_ratio = class5_counter/float(dataset_size)
    class6_occurrence_ratio = class6_counter/float(dataset_size)
    class7_occurrence_ratio = class7_counter/float(dataset_size)
    class8_occurrence_ratio = class8_counter/float(dataset_size)
    class9_occurrence_ratio = class9_counter/float(dataset_size)
    class10_occurrence_ratio = class10_counter/float(dataset_size)

    # FEATURE STOPWORD FREQUENCY
    class1_stopwords_frequency_arr = []
    class2_stopwords_frequency_arr = []
    class3_stopwords_frequency_arr = []
    class4_stopwords_frequency_arr = []
    class5_stopwords_frequency_arr = []
    class6_stopwords_frequency_arr = []
    class7_stopwords_frequency_arr = []
    class8_stopwords_frequency_arr = []
    class9_stopwords_frequency_arr = []
    class10_stopwords_frequency_arr = []
    # get stopword frequencies, calculate mean and stdeviation
    i = 0
    while i < 9:
        class1_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[i][1]))
        class2_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[10+i][1]))
        class3_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[20+i][1]))
        class4_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[30+i][1]))
        class5_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[40+i][1]))
        class6_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[50+i][1]))
        class7_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[60+i][1]))
        class8_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[70+i][1]))
        class9_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[80+i][1]))
        class10_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[90+i][1]))
        i += 1

    # class 1
    class1_stopwords_frequency_mean = mean(
        class1_stopwords_frequency_arr)
    class1_stopwords_frequency_stdev = stdev(
        class1_stopwords_frequency_arr)
    # class2
    class2_stopwords_frequency_mean = mean(
        class2_stopwords_frequency_arr)
    class2_stopwords_frequency_stdev = stdev(
        class2_stopwords_frequency_arr)
    # class3
    class3_stopwords_frequency_mean = mean(
        class3_stopwords_frequency_arr)
    class3_stopwords_frequency_stdev = stdev(
        class3_stopwords_frequency_arr)
    # class4
    class4_stopwords_frequency_mean = mean(
        class4_stopwords_frequency_arr)
    class4_stopwords_frequency_stdev = stdev(
        class4_stopwords_frequency_arr)
    # class5
    class5_stopwords_frequency_mean = mean(
        class5_stopwords_frequency_arr)
    class5_stopwords_frequency_stdev = stdev(
        class5_stopwords_frequency_arr)
    # class6
    class6_stopwords_frequency_mean = mean(
        class6_stopwords_frequency_arr)
    class6_stopwords_frequency_stdev = stdev(
        class6_stopwords_frequency_arr)
    # class7
    class7_stopwords_frequency_mean = mean(
        class7_stopwords_frequency_arr)
    class7_stopwords_frequency_stdev = stdev(
        class7_stopwords_frequency_arr)
    # class8
    class8_stopwords_frequency_mean = mean(
        class8_stopwords_frequency_arr)
    class8_stopwords_frequency_stdev = stdev(
        class8_stopwords_frequency_arr)
    # class9
    class9_stopwords_frequency_mean = mean(
        class9_stopwords_frequency_arr)
    class9_stopwords_frequency_stdev = stdev(
        class9_stopwords_frequency_arr)
    # class10
    class10_stopwords_frequency_mean = mean(
        class10_stopwords_frequency_arr)
    class10_stopwords_frequency_stdev = stdev(
        class10_stopwords_frequency_arr)

    # FEATURE AVERAGE SENTENCE LENGTH
    class1_average_sentence_length_arr = []
    class2_average_sentence_length_arr = []
    class3_average_sentence_length_arr = []
    class4_average_sentence_length_arr = []
    class5_average_sentence_length_arr = []
    class6_average_sentence_length_arr = []
    class7_average_sentence_length_arr = []
    class8_average_sentence_length_arr = []
    class9_average_sentence_length_arr = []
    class10_average_sentence_length_arr = []

    i = 0
    while i < 9:
        class1_average_sentence_length_arr.append(
            average_sentence_length(dataset[i][1]))
        class2_average_sentence_length_arr.append(
            average_sentence_length(dataset[10+i][1]))
        class3_average_sentence_length_arr.append(
            average_sentence_length(dataset[20+i][1]))
        class4_average_sentence_length_arr.append(
            average_sentence_length(dataset[30+i][1]))
        class5_average_sentence_length_arr.append(
            average_sentence_length(dataset[40+i][1]))
        class6_average_sentence_length_arr.append(
            average_sentence_length(dataset[50+i][1]))
        class7_average_sentence_length_arr.append(
            average_sentence_length(dataset[60+i][1]))
        class8_average_sentence_length_arr.append(
            average_sentence_length(dataset[70+i][1]))
        class9_average_sentence_length_arr.append(
            average_sentence_length(dataset[80+i][1]))
        class10_average_sentence_length_arr.append(
            average_sentence_length(dataset[90+i][1]))
        i += 1

    # class 1
    class1_average_sentence_length_mean = mean(
        class1_average_sentence_length_arr)
    class1_average_sentence_length_stdev = stdev(
        class1_average_sentence_length_arr)
    # class2
    class2_average_sentence_length_mean = mean(
        class2_average_sentence_length_arr)
    class2_average_sentence_length_stdev = stdev(
        class2_average_sentence_length_arr)
    # class3
    class3_average_sentence_length_mean = mean(
        class3_average_sentence_length_arr)
    class3_average_sentence_length_stdev = stdev(
        class3_average_sentence_length_arr)
    # class4
    class4_average_sentence_length_mean = mean(
        class4_average_sentence_length_arr)
    class4_average_sentence_length_stdev = stdev(
        class4_average_sentence_length_arr)
    # class5
    class5_average_sentence_length_mean = mean(
        class5_average_sentence_length_arr)
    class5_average_sentence_length_stdev = stdev(
        class5_average_sentence_length_arr)
    # class6
    class6_average_sentence_length_mean = mean(
        class6_average_sentence_length_arr)
    class6_average_sentence_length_stdev = stdev(
        class6_average_sentence_length_arr)
    # class7
    class7_average_sentence_length_mean = mean(
        class7_average_sentence_length_arr)
    class7_average_sentence_length_stdev = stdev(
        class7_average_sentence_length_arr)
    # class8
    class8_average_sentence_length_mean = mean(
        class8_average_sentence_length_arr)
    class8_average_sentence_length_stdev = stdev(
        class8_average_sentence_length_arr)
    # class9
    class9_average_sentence_length_mean = mean(
        class9_average_sentence_length_arr)
    class9_average_sentence_length_stdev = stdev(
        class9_average_sentence_length_arr)
    # class10
    class10_average_sentence_length_mean = mean(
        class10_average_sentence_length_arr)
    class10_average_sentence_length_stdev = stdev(
        class10_average_sentence_length_arr)

    # FEATURE SENTENCE DISTRIBUTION
    # SIMPLE SENTENCE DISTRIBUTION
    class1_simple_sentence_distribution_arr = []
    class2_simple_sentence_distribution_arr = []
    class3_simple_sentence_distribution_arr = []
    class4_simple_sentence_distribution_arr = []
    class5_simple_sentence_distribution_arr = []
    class6_simple_sentence_distribution_arr = []
    class7_simple_sentence_distribution_arr = []
    class8_simple_sentence_distribution_arr = []
    class9_simple_sentence_distribution_arr = []
    class10_simple_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[0])
        class2_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[10+i][1])[0])
        class3_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[0])
        class4_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[30+i][1])[0])
        class5_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[40+i][1])[0])
        class6_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[50+i][1])[0])
        class7_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[60+i][1])[0])
        class8_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[70+i][1])[0])
        class9_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[80+i][1])[0])
        class10_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[90+i][1])[0])
        i += 1

     # class 1
    class1_simple_sentence_distribution_mean = mean(
        class1_simple_sentence_distribution_arr)
    class1_simple_sentence_distribution_stdev = stdev(
        class1_simple_sentence_distribution_arr)
    # class2
    class2_simple_sentence_distribution_mean = mean(
        class2_simple_sentence_distribution_arr)
    class2_simple_sentence_distribution_stdev = stdev(
        class2_simple_sentence_distribution_arr)
    # class3
    class3_simple_sentence_distribution_mean = mean(
        class3_simple_sentence_distribution_arr)
    class3_simple_sentence_distribution_stdev = stdev(
        class3_simple_sentence_distribution_arr)
    # class4
    class4_simple_sentence_distribution_mean = mean(
        class4_simple_sentence_distribution_arr)
    class4_simple_sentence_distribution_stdev = stdev(
        class4_simple_sentence_distribution_arr)
    # class5
    class5_simple_sentence_distribution_mean = mean(
        class5_simple_sentence_distribution_arr)
    class5_simple_sentence_distribution_stdev = stdev(
        class5_simple_sentence_distribution_arr)
    # class6
    class6_simple_sentence_distribution_mean = mean(
        class6_simple_sentence_distribution_arr)
    class6_simple_sentence_distribution_stdev = stdev(
        class6_simple_sentence_distribution_arr)
    # class7
    class7_simple_sentence_distribution_mean = mean(
        class7_simple_sentence_distribution_arr)
    class7_simple_sentence_distribution_stdev = stdev(
        class7_simple_sentence_distribution_arr)
    # class8
    class8_simple_sentence_distribution_mean = mean(
        class8_simple_sentence_distribution_arr)
    class8_simple_sentence_distribution_stdev = stdev(
        class8_simple_sentence_distribution_arr)
    # class9
    class9_simple_sentence_distribution_mean = mean(
        class9_simple_sentence_distribution_arr)
    class9_simple_sentence_distribution_stdev = stdev(
        class9_simple_sentence_distribution_arr)
    # class10
    class10_simple_sentence_distribution_mean = mean(
        class10_simple_sentence_distribution_arr)
    class10_simple_sentence_distribution_stdev = stdev(
        class10_simple_sentence_distribution_arr)

    # COMPOUND SENTENCE DISTRIBUTION
    class1_compound_sentence_distribution_arr = []
    class2_compound_sentence_distribution_arr = []
    class3_compound_sentence_distribution_arr = []
    class4_compound_sentence_distribution_arr = []
    class5_compound_sentence_distribution_arr = []
    class6_compound_sentence_distribution_arr = []
    class7_compound_sentence_distribution_arr = []
    class8_compound_sentence_distribution_arr = []
    class9_compound_sentence_distribution_arr = []
    class10_compound_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[1])
        class2_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[10+i][1])[1])
        class3_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[1])
        class4_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[30+i][1])[1])
        class5_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[40+i][1])[1])
        class6_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[50+i][1])[1])
        class7_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[60+i][1])[1])
        class8_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[70+i][1])[1])
        class9_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[80+i][1])[1])
        class10_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[90+i][1])[1])
        i += 1
     # class 1
    class1_compound_sentence_distribution_stdev = stdev(
        class1_compound_sentence_distribution_arr)
    class1_compound_sentence_distribution_mean = mean(
        class1_compound_sentence_distribution_arr)
    # class2
    class2_compound_sentence_distribution_mean = mean(
        class2_compound_sentence_distribution_arr)
    class2_compound_sentence_distribution_stdev = stdev(
        class2_compound_sentence_distribution_arr)
    # class3
    class3_compound_sentence_distribution_mean = mean(
        class3_compound_sentence_distribution_arr)
    class3_compound_sentence_distribution_stdev = stdev(
        class3_compound_sentence_distribution_arr)
    # class4
    class4_compound_sentence_distribution_mean = mean(
        class4_compound_sentence_distribution_arr)
    class4_compound_sentence_distribution_stdev = stdev(
        class4_compound_sentence_distribution_arr)
    # class5
    class5_compound_sentence_distribution_mean = mean(
        class5_compound_sentence_distribution_arr)
    class5_compound_sentence_distribution_stdev = stdev(
        class5_compound_sentence_distribution_arr)
    # class6
    class6_compound_sentence_distribution_mean = mean(
        class6_compound_sentence_distribution_arr)
    class6_compound_sentence_distribution_stdev = stdev(
        class6_compound_sentence_distribution_arr)
    # class7
    class7_compound_sentence_distribution_mean = mean(
        class7_compound_sentence_distribution_arr)
    class7_compound_sentence_distribution_stdev = stdev(
        class7_compound_sentence_distribution_arr)
    # class8
    class8_compound_sentence_distribution_mean = mean(
        class8_compound_sentence_distribution_arr)
    class8_compound_sentence_distribution_stdev = stdev(
        class8_compound_sentence_distribution_arr)
    # class9
    class9_compound_sentence_distribution_mean = mean(
        class9_compound_sentence_distribution_arr)
    class9_compound_sentence_distribution_stdev = stdev(
        class9_compound_sentence_distribution_arr)
    # class10
    class10_compound_sentence_distribution_mean = mean(
        class10_compound_sentence_distribution_arr)
    class10_compound_sentence_distribution_stdev = stdev(
        class10_compound_sentence_distribution_arr)

    # COMPLEX SENTENCE DISTRIBUTION
    class1_complex_sentence_distribution_arr = []
    class2_complex_sentence_distribution_arr = []
    class3_complex_sentence_distribution_arr = []
    class4_complex_sentence_distribution_arr = []
    class5_complex_sentence_distribution_arr = []
    class6_complex_sentence_distribution_arr = []
    class7_complex_sentence_distribution_arr = []
    class8_complex_sentence_distribution_arr = []
    class9_complex_sentence_distribution_arr = []
    class10_complex_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[2])
        class2_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[10+i][1])[2])
        class3_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[2])
        class4_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[30+i][1])[2])
        class5_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[40+i][1])[2])
        class6_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[50+i][1])[2])
        class7_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[60+i][1])[2])
        class8_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[70+i][1])[2])
        class9_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[80+i][1])[2])
        class10_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[90+i][1])[2])
        i += 1
     # class 1
    class1_complex_sentence_distribution_stdev = stdev(
        class1_complex_sentence_distribution_arr)
    class1_complex_sentence_distribution_mean = mean(
        class1_complex_sentence_distribution_arr)
    # class2
    class2_complex_sentence_distribution_mean = mean(
        class2_complex_sentence_distribution_arr)
    class2_complex_sentence_distribution_stdev = stdev(
        class2_complex_sentence_distribution_arr)
    # class3
    class3_complex_sentence_distribution_mean = mean(
        class3_complex_sentence_distribution_arr)
    class3_complex_sentence_distribution_stdev = stdev(
        class3_complex_sentence_distribution_arr)
    # class4
    class4_complex_sentence_distribution_mean = mean(
        class4_complex_sentence_distribution_arr)
    class4_complex_sentence_distribution_stdev = stdev(
        class4_complex_sentence_distribution_arr)
    # class5
    class5_complex_sentence_distribution_mean = mean(
        class5_complex_sentence_distribution_arr)
    class5_complex_sentence_distribution_stdev = stdev(
        class5_complex_sentence_distribution_arr)
    # class6
    class6_complex_sentence_distribution_mean = mean(
        class6_complex_sentence_distribution_arr)
    class6_complex_sentence_distribution_stdev = stdev(
        class6_complex_sentence_distribution_arr)
    # class7
    class7_complex_sentence_distribution_mean = mean(
        class7_complex_sentence_distribution_arr)
    class7_complex_sentence_distribution_stdev = stdev(
        class7_complex_sentence_distribution_arr)
    # class8
    class8_complex_sentence_distribution_mean = mean(
        class8_complex_sentence_distribution_arr)
    class8_complex_sentence_distribution_stdev = stdev(
        class8_complex_sentence_distribution_arr)
    # class9
    class9_complex_sentence_distribution_mean = mean(
        class9_complex_sentence_distribution_arr)
    class9_complex_sentence_distribution_stdev = stdev(
        class9_complex_sentence_distribution_arr)
    # class10
    class10_complex_sentence_distribution_mean = mean(
        class10_complex_sentence_distribution_arr)
    class10_complex_sentence_distribution_stdev = stdev(
        class10_complex_sentence_distribution_arr)

    class1_stats = {'stopwordsFrequency': {'mean': class1_stopwords_frequency_mean, 'stdev': class1_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class1_simple_sentence_distribution_mean, 'stdev': class1_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class1_compound_sentence_distribution_mean, 'stdev': class1_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class1_complex_sentence_distribution_mean, 'stdev': class1_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class1_average_sentence_length_mean, 'stdev': class1_average_sentence_length_stdev}}
    class2_stats = {'stopwordsFrequency': {'mean': class2_stopwords_frequency_mean, 'stdev': class2_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class2_simple_sentence_distribution_mean, 'stdev': class2_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class2_compound_sentence_distribution_mean, 'stdev': class2_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class2_complex_sentence_distribution_mean, 'stdev': class2_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class2_average_sentence_length_mean, 'stdev': class2_average_sentence_length_stdev}}
    class3_stats = {'stopwordsFrequency': {'mean': class3_stopwords_frequency_mean, 'stdev': class3_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class3_simple_sentence_distribution_mean, 'stdev': class3_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class3_compound_sentence_distribution_mean, 'stdev': class3_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class3_complex_sentence_distribution_mean, 'stdev': class3_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class3_average_sentence_length_mean, 'stdev': class3_average_sentence_length_stdev}}
    class4_stats = {'stopwordsFrequency': {'mean': class4_stopwords_frequency_mean, 'stdev': class4_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class4_simple_sentence_distribution_mean, 'stdev': class4_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class4_compound_sentence_distribution_mean, 'stdev': class4_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class4_complex_sentence_distribution_mean, 'stdev': class4_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class4_average_sentence_length_mean, 'stdev': class4_average_sentence_length_stdev}}
    class5_stats = {'stopwordsFrequency': {'mean': class5_stopwords_frequency_mean, 'stdev': class5_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class5_simple_sentence_distribution_mean, 'stdev': class5_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class5_compound_sentence_distribution_mean, 'stdev': class5_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class5_complex_sentence_distribution_mean, 'stdev': class5_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class5_average_sentence_length_mean, 'stdev': class5_average_sentence_length_stdev}}
    class6_stats = {'stopwordsFrequency': {'mean': class6_stopwords_frequency_mean, 'stdev': class6_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class6_simple_sentence_distribution_mean, 'stdev': class6_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class6_compound_sentence_distribution_mean, 'stdev': class6_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class6_complex_sentence_distribution_mean, 'stdev': class6_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class6_average_sentence_length_mean, 'stdev': class6_average_sentence_length_stdev}}
    class7_stats = {'stopwordsFrequency': {'mean': class7_stopwords_frequency_mean, 'stdev': class7_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class7_simple_sentence_distribution_mean, 'stdev': class7_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class7_compound_sentence_distribution_mean, 'stdev': class7_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class7_complex_sentence_distribution_mean, 'stdev': class7_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class7_average_sentence_length_mean, 'stdev': class7_average_sentence_length_stdev}}
    class8_stats = {'stopwordsFrequency': {'mean': class8_stopwords_frequency_mean, 'stdev': class8_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class8_simple_sentence_distribution_mean, 'stdev': class8_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class8_compound_sentence_distribution_mean, 'stdev': class8_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class8_complex_sentence_distribution_mean, 'stdev': class8_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class8_average_sentence_length_mean, 'stdev': class8_average_sentence_length_stdev}}
    class9_stats = {'stopwordsFrequency': {'mean': class9_stopwords_frequency_mean, 'stdev': class9_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class9_simple_sentence_distribution_mean, 'stdev': class9_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class9_compound_sentence_distribution_mean, 'stdev': class9_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class9_complex_sentence_distribution_mean, 'stdev': class9_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class9_average_sentence_length_mean, 'stdev': class9_average_sentence_length_stdev}}
    class10_stats = {'stopwordsFrequency': {'mean': class10_stopwords_frequency_mean, 'stdev': class10_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class10_simple_sentence_distribution_mean, 'stdev': class10_simple_sentence_distribution_stdev},
                     'compoundSentenceDistribution': {'mean': class10_compound_sentence_distribution_mean, 'stdev': class10_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class10_complex_sentence_distribution_mean, 'stdev': class10_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class10_average_sentence_length_mean, 'stdev': class10_average_sentence_length_stdev}}
    import json
    data = {}
    # data['classData']
    data['classData'] = {
        "class1_stats": class1_stats,
        "class2_stats": class2_stats,
        "class3_stats": class3_stats,
        "class4_stats": class4_stats,
        "class5_stats": class5_stats,
        "class6_stats": class6_stats,
        "class7_stats": class7_stats,
        "class8_stats": class8_stats,
        "class9_stats": class9_stats,
        "class10_stats": class10_stats,
        "class1_occurrence_ratio": class1_occurrence_ratio,
        "class2_occurrence_ratio": class2_occurrence_ratio,
        "class3_occurrence_ratio": class3_occurrence_ratio,
        "class4_occurrence_ratio": class4_occurrence_ratio,
        "class5_occurrence_ratio": class5_occurrence_ratio,
        "class6_occurrence_ratio": class6_occurrence_ratio,
        "class7_occurrence_ratio": class7_occurrence_ratio,
        "class8_occurrence_ratio": class8_occurrence_ratio,
        "class9_occurrence_ratio": class9_occurrence_ratio,
        "class10_occurrence_ratio": class10_occurrence_ratio,
    }
    with open('trainedModel.json', 'w') as outfile:
        json.dump(data, outfile)
    return([class1_stats, class2_stats, class3_stats, class4_stats, class5_stats, class6_stats, class7_stats, class8_stats, class9_stats, class10_stats, class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio, class4_occurrence_ratio, class5_occurrence_ratio, class6_occurrence_ratio, class7_occurrence_ratio, class8_occurrence_ratio, class9_occurrence_ratio, class10_occurrence_ratio])


def calculate_probability(x, mean, stdev):
    var = float(stdev)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# calculate standard deviation


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)


@app.route('/author_id/api/predict_class', methods=['POST'])
def predict_class():
    req_data = request.get_json()
    text = req_data['raw_text']
    # print(req_data['raw_text'])
    # model = naive_bayes()
    import json
    with open('trainedModel.json') as json_file:
        data = json.load(json_file)
    print(data["classData"]["class1_stats"])
    class1_stats = data["classData"]["class1_stats"]
    class2_stats = data["classData"]["class2_stats"]
    class3_stats = data["classData"]["class3_stats"]
    class4_stats = data["classData"]["class4_stats"]
    class5_stats = data["classData"]["class5_stats"]
    class6_stats = data["classData"]["class6_stats"]
    class7_stats = data["classData"]["class7_stats"]
    class8_stats = data["classData"]["class8_stats"]
    class9_stats = data["classData"]["class9_stats"]
    class10_stats = data["classData"]["class10_stats"]

    class1_occurrence_ratio = data["classData"]["class1_occurrence_ratio"]
    class2_occurrence_ratio = data["classData"]["class2_occurrence_ratio"]
    class3_occurrence_ratio = data["classData"]["class3_occurrence_ratio"]
    class4_occurrence_ratio = data["classData"]["class4_occurrence_ratio"]
    class5_occurrence_ratio = data["classData"]["class5_occurrence_ratio"]
    class6_occurrence_ratio = data["classData"]["class6_occurrence_ratio"]
    class7_occurrence_ratio = data["classData"]["class7_occurrence_ratio"]
    class8_occurrence_ratio = data["classData"]["class8_occurrence_ratio"]
    class9_occurrence_ratio = data["classData"]["class9_occurrence_ratio"]
    class10_occurrence_ratio = data["classData"]["class10_occurrence_ratio"]

    # stopwords freq
    stopwords_frequency = stopwords_frequency_ratio(text)
    # average sentence length
    average_sentenceLength = average_sentence_length(text)
    # simple sentence distribution
    simple_sentence_distribution = sentence_length_distribution(text)[0]
    # compound sentence distribution
    compound_sentence_distribution = sentence_length_distribution(text)[1]
    # complex sentence distribution
    complex_sentence_distribution = sentence_length_distribution(text)[2]

    # get probabilities and compare classes
    # class1
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class1_stats['stopwordsFrequency']['mean'], class1_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class1_stats['averageSentencelength']['mean'], class1_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class1_stats['simpleSentenceDistribution']['mean'], class1_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class1_stats['compoundSentenceDistribution']['mean'], class1_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class1_stats['complexSentenceDistribution']['mean'], class1_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class1_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution)*class1_occurrence_ratio
    # print(class1_probability)
    # class2
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class2_stats['stopwordsFrequency']['mean'], class2_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class2_stats['averageSentencelength']['mean'], class2_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class2_stats['simpleSentenceDistribution']['mean'], class2_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class2_stats['compoundSentenceDistribution']['mean'], class2_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class2_stats['complexSentenceDistribution']['mean'], class2_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class2_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class2_occurrence_ratio
    # print(class2_probability)
    # class3
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class3_stats['stopwordsFrequency']['mean'], class3_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class3_stats['averageSentencelength']['mean'], class3_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class3_stats['simpleSentenceDistribution']['mean'], class3_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class3_stats['compoundSentenceDistribution']['mean'], class3_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class3_stats['complexSentenceDistribution']['mean'], class3_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class3_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class3_occurrence_ratio
    # print(class3_probability)
    # class4
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class4_stats['stopwordsFrequency']['mean'], class4_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class4_stats['averageSentencelength']['mean'], class4_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class4_stats['simpleSentenceDistribution']['mean'], class4_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class4_stats['compoundSentenceDistribution']['mean'], class4_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class4_stats['complexSentenceDistribution']['mean'], class4_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class4_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class4_occurrence_ratio
    # print(class4_probability)
    # class5
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class5_stats['stopwordsFrequency']['mean'], class5_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class5_stats['averageSentencelength']['mean'], class5_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class5_stats['simpleSentenceDistribution']['mean'], class5_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class5_stats['compoundSentenceDistribution']['mean'], class5_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class5_stats['complexSentenceDistribution']['mean'], class5_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class5_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class5_occurrence_ratio
    # print(class5_probability)
    # class6
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class6_stats['stopwordsFrequency']['mean'], class6_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class6_stats['averageSentencelength']['mean'], class6_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class6_stats['simpleSentenceDistribution']['mean'], class6_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class6_stats['compoundSentenceDistribution']['mean'], class6_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class6_stats['complexSentenceDistribution']['mean'], class6_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class6_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class6_occurrence_ratio
    # print(class6_probability)
    # class7
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class7_stats['stopwordsFrequency']['mean'], class7_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class7_stats['averageSentencelength']['mean'], class7_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class7_stats['simpleSentenceDistribution']['mean'], class7_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class7_stats['compoundSentenceDistribution']['mean'], class7_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class7_stats['complexSentenceDistribution']['mean'], class7_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class7_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class7_occurrence_ratio
    # print(class7_probability)
    # class8
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class8_stats['stopwordsFrequency']['mean'], class8_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class8_stats['averageSentencelength']['mean'], class8_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class8_stats['simpleSentenceDistribution']['mean'], class8_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class8_stats['compoundSentenceDistribution']['mean'], class8_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class8_stats['complexSentenceDistribution']['mean'], class8_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class8_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class8_occurrence_ratio
    # print(class8_probability)
    # class9
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class9_stats['stopwordsFrequency']['mean'], class9_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class9_stats['averageSentencelength']['mean'], class9_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class9_stats['simpleSentenceDistribution']['mean'], class9_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class9_stats['compoundSentenceDistribution']['mean'], class9_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class9_stats['complexSentenceDistribution']['mean'], class9_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class9_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                          probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class9_occurrence_ratio
    # print(class9_probability)
    # class10
    probability_stopwords_frequency = calculate_probability(
        stopwords_frequency, class10_stats['stopwordsFrequency']['mean'], class10_stats['stopwordsFrequency']['stdev'])
    probability_average_sentenceLength = calculate_probability(
        average_sentenceLength, class10_stats['averageSentencelength']['mean'], class10_stats['averageSentencelength']['stdev'])
    probability_simple_sentence_distribution = calculate_probability(
        simple_sentence_distribution, class10_stats['simpleSentenceDistribution']['mean'], class10_stats['simpleSentenceDistribution']['stdev'])
    probability_compound_sentence_distribution = calculate_probability(
        compound_sentence_distribution, class10_stats['compoundSentenceDistribution']['mean'], class10_stats['compoundSentenceDistribution']['stdev'])
    probability_complex_sentence_distribution = calculate_probability(
        complex_sentence_distribution, class10_stats['complexSentenceDistribution']['mean'], class10_stats['complexSentenceDistribution']['stdev'])
    # print(probability_stopwords_frequency, probability_simple_sentence_distribution,
    #       probability_compound_sentence_distribution, probability_complex_sentence_distribution)
    class10_probability = (probability_stopwords_frequency*probability_average_sentenceLength*probability_simple_sentence_distribution *
                           probability_compound_sentence_distribution*probability_complex_sentence_distribution) * class10_occurrence_ratio
    # print(class10_probability)
    classProbabilities = [class1_probability, class2_probability, class3_probability, class4_probability, class5_probability,
                          class6_probability, class7_probability, class8_probability, class9_probability, class10_probability]
    predictedClass = ''
    maxProbability = max(classProbabilities)
    predictedClassIndexes = [
        i for i, j in enumerate(classProbabilities) if j == maxProbability]
    labels = ['Faith Oneya', 'Bitange Ndemo', 'Abigail Arunga', 'Walter Menya', 'Ruth Mbula',
              'David Mwere', 'Justus Ochieng', 'Sam Kiplagat', 'Samuel Owino' 'Joseph Mboya']

    # if(class1_probability > class2_probability and class1_probability > class3_probability):
    #     print('Predicted Class 1 - Faith Oneya')
    #     predictedClass = 'Predicted Class 1 - Faith Oneya'
    # elif(class2_probability > class1_probability and class2_probability > class3_probability):
    #     print('Predicted Class 2 - Bitange Ndemo')
    #     predictedClass = 'Predicted Class 2 - Bitange Ndemo'
    # elif(class3_probability > class1_probability and class3_probability > class2_probability):
    #     print('Predicted Class 3 - Abigail Arunga')
    #     predictedClass = 'Predicted Class 3 - Abigail Arunga'
    # return predictedClass
    return jsonify({"predicted_class": labels[predictedClassIndexes[0]],  # stopwords freq
                    "stopwords_frequency": stopwords_frequency,
                    # average sentence length
                    "average_sentenceLength": average_sentenceLength,
                    # simple sentence distribution
                    "simple_sentence_distribution": simple_sentence_distribution,
                    # compound sentence distribution
                    "compound_sentence_distribution": compound_sentence_distribution,
                    # complex sentence distribution
                    "complex_sentence_distribution": complex_sentence_distribution,
                    "class1_probability": class1_probability,
                    "class2_probability": class2_probability,
                    "class3_probability": class3_probability,
                    "class4_probability": class4_probability,
                    "class5_probability": class5_probability,
                    "class6_probability": class6_probability,
                    "class7_probability": class7_probability,
                    "class8_probability": class8_probability,
                    "class9_probability": class9_probability,
                    "class10_probability": class10_probability,
                    }), 200

    # print(sentence_length_distribution('When it’s not an integer, the highest probability number of events will be the nearest integer to the rate parameter, since the Poisson distribution is only defined for a discrete number of events. The discrete nature of the Poisson distribution is also why this is a probability mass function and not a density function. We can use the Poisson Distribution mass function to find the probability of observing a number of events over an interval generated by a Poisson process. Another use of the mass function equation — as we’ll see later — is to find the probability of waiting some time between events. For the problem we’ll solve with a Poisson distribution, we could continue with website failures, but I propose something grander. In my childhood, my father would often take me into our yard to observe (or try to observe) meteor showers. We were not space geeks, but watching objects from outer space burn up in the sky was enough to get us outside even though meteor showers always seemed to occur in the coldest months. The number of meteors seen can be modeled as a Poisson distribution because the meteors are independent, the average number of meteors per hour is constant (in the short term), and this is an approximation meteors don’t occur simultaneously. To characterize the Poisson distribution, all we need is the rate parameter which is the number of events/interval * interval length. From what I remember, we were told to expect 5 meteors per hour on average or 1 every 12 minutes. Due to the limited patience of a young child (especially on a freezing night), we never stayed out more than 60 minutes, so we’ll use that as the time period. Putting the two together, we get:'))
# naive_bayes()
# print(calculate_probability(1.0,1.0,1.0))
# ]
# predict_class('Faced with the daunting reality of empty accounts post-Covid-19, private schools have been forced to take control of their financial fate by reinventing their income streams at the expense of parents and learners. With all the attention focused on the war against Covid-19, most schools have taken the opportunity to make outrageous demands in terms of school fees, safe in the knowledge that their actions might barely be noticed. To be fair, the pandemic has forced some hard choices on schools, which were first to be closed on March 15 as they are considered high-risk areas because of the many learners they hold, and who come from different backgrounds. The Daily Nation recently carried a story showing how fancy schools are struggling to stay afloat. Nevertheless, this doesnâ€™t negate the fact that some of the decisions taken by schools â€” like rushing to charge them astronomical amounts for virtual learning â€” have harmed parents. When Brookhouse School demanded a 90 per cent school fees payment from parents for online learning, they fought back by suing the school and the court compelled the institution to reduce the fees by half. Kudos to the parents. While one canâ€™t extrapolate too much on the case of the Brookhouse School parents, given that they are not a towering example of the average Kenyan parent, itâ€™s still a valid example of the frustrations most parents are feeling with their childrenâ€™s schools.SCARCE RESOURCES The proponents of the Competency-Based Curriculum, to which a majority of schools in Kenya subscribe, boast about the level of involvement from a parent thatâ€™s needed for the learner to succeed but the top-down approach to imposing supplementary virtual learning classes on parents is the antithesis of this. Most parents will testify that nobody asked them what they needed and what would work for their children, schedules or homes.The notion of synchronous virtual learning, which most schools have suggested to or imposed on parents, is as utopian as it is unreasonable, given the limited computer access in homes as well as spotty WiFi, among other obstacles like the expenses involved and the crazy work schedules that parents who work in essential services have been forced to adapt to because of the pandemic. Kenya National Union of Teachers Secretary-General Wilson Sossion has pointed out that the practicability of virtual lessons is far from being a reality in Kenya, adding that supervision in a manner thatâ€™s useful to the students is also an issue. Despite much hype by the government and private schools about virtual learning, one of the things they fail to acknowledge is that it will leave out a significant proportion of learners. It will be survival for the wealthiest, not the fittest.Data from the Kenya National Bureau of Statistics (KNBS) revealed that the digital divide in Kenya is still a reality, as only one in five Kenyans have access to the internet. BLANKET SOLUTIONS The income of the parents also determines whether their children will access virtual lessons or not. The National ICT Survey Report (2018) by the Communication Authority and the KNBS indicated that employment and household size in many cases determine the household disposable income, which in turn determines whether individuals in households can afford radios, TVs, computers, internet and other ICT equipment. Whatâ€™s clear is that by offering blanket solutions for virtual learning, private schools are just combing a dry riverbed. There is an opportunity here to respond to the abnormal times abnormally, to paraphrase Health Cabinet Secretary Mutahi Kagweâ€™s now-famous quote about Covid-19. In their quest to keep afloat, private schools should not forget to fully involve parents in designing solutions that will accommodate different learnerâ€™s and parentsâ€™ needs. Nobody will walk away from this pandemic unscathed, but as everything comes unglued after this, private schools can help lessen the impact by doing right by parents and learners')
