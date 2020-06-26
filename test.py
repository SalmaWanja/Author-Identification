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
    sentences = nltk.tokenize.sent_tokenize(text.decode('utf-8').strip())
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
    sentences = nltk.tokenize.sent_tokenize(text.decode('utf-8').strip())
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
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3}
    dataset = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                dataset.append([labels[row[0]], row[1]])
                line_count += 1
    return dataset


def naive_bayes():
    dataset = load_dataset('myDataset.csv')
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
        # average_sentence_length(dataset[i][1]))
        class2_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[10+i][1]))
        # average_sentence_length(dataset[10+i][1]))
        class3_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[20+i][1]))
        # average_sentence_length(dataset[20+i][1]))
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

    # FEATURE AVERAGE SENTENCE LENGTH
    class1_average_sentence_length_arr = []
    class2_average_sentence_length_arr = []
    class3_average_sentence_length_arr = []

    i = 0
    while i < 9:
        class1_average_sentence_length_arr.append(
            average_sentence_length(dataset[i][1]))
        class2_average_sentence_length_arr.append(
            average_sentence_length(dataset[10+i][1]))
        class3_average_sentence_length_arr.append(
            average_sentence_length(dataset[20+i][1]))
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
            sentence_length_distribution(dataset[10+i][1])[0])
        class3_simple_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[0])
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

    # COMPOUND SENTENCE DISTRIBUTION
    class1_compound_sentence_distribution_arr = []
    class2_compound_sentence_distribution_arr = []
    class3_compound_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[1])
        class2_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[10+i][1])[1])
        class3_compound_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[1])
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

    # COMPLEX SENTENCE DISTRIBUTION
    class1_complex_sentence_distribution_arr = []
    class2_complex_sentence_distribution_arr = []
    class3_complex_sentence_distribution_arr = []

    i = 0
    while i < 9:
        class1_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[i][1])[2])
        class2_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[10+i][1])[2])
        class3_complex_sentence_distribution_arr.append(
            sentence_length_distribution(dataset[20+i][1])[2])
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

    class1_stats = {'stopwordsFrequency': {'mean': class1_stopwords_frequency_mean, 'stdev': class1_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class1_simple_sentence_distribution_mean, 'stdev': class1_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class1_compound_sentence_distribution_mean, 'stdev': class1_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class1_complex_sentence_distribution_mean, 'stdev': class1_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class1_average_sentence_length_mean, 'stdev': class1_average_sentence_length_stdev}}
    class2_stats = {'stopwordsFrequency': {'mean': class2_stopwords_frequency_mean, 'stdev': class2_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class2_simple_sentence_distribution_mean, 'stdev': class2_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class2_compound_sentence_distribution_mean, 'stdev': class2_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class2_complex_sentence_distribution_mean, 'stdev': class2_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class2_average_sentence_length_mean, 'stdev': class2_average_sentence_length_stdev}}
    class3_stats = {'stopwordsFrequency': {'mean': class3_stopwords_frequency_mean, 'stdev': class3_stopwords_frequency_stdev}, 'simpleSentenceDistribution': {'mean': class3_simple_sentence_distribution_mean, 'stdev': class3_simple_sentence_distribution_stdev},
                    'compoundSentenceDistribution': {'mean': class3_compound_sentence_distribution_mean, 'stdev': class3_compound_sentence_distribution_stdev}, 'complexSentenceDistribution': {'mean': class3_complex_sentence_distribution_mean, 'stdev': class3_complex_sentence_distribution_stdev}, 'averageSentencelength': {'mean': class3_average_sentence_length_mean, 'stdev': class3_average_sentence_length_stdev}}

    # test_classifier(class1_stats, class2_stats, class3_stats,
    #                 class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio)
    # text = "Faced with the daunting reality of empty accounts post-Covid-19, private schools have been forced to take control of their financial fate by reinventing their income streams at the expense of parents and learners. With all the attention focused on the war against Covid-19, most schools have taken the opportunity to make outrageous demands in terms of school fees, safe in the knowledge that their actions might barely be noticed. To be fair, the pandemic has forced some hard choices on schools, which were first to be closed on March 15 as they are considered high-risk areas because of the many learners they hold, and who come from different backgrounds. The Daily Nation recently carried a story showing how fancy schools are struggling to stay afloat. Nevertheless, this doesnâ€™t negate the fact that some of the decisions taken by schools â€” like rushing to charge them astronomical amounts for virtual learning â€” have harmed parents. When Brookhouse School demanded a 90 per cent school fees payment from parents for online learning, they fought back by suing the school and the court compelled the institution to reduce the fees by half. Kudos to the parents. While one canâ€™t extrapolate too much on the case of the Brookhouse School parents, given that they are not a towering example of the average Kenyan parent, itâ€™s still a valid example of the frustrations most parents are feeling with their childrenâ€™s schools.SCARCE RESOURCES The proponents of the Competency-Based Curriculum, to which a majority of schools in Kenya subscribe, boast about the level of involvement from a parent thatâ€™s needed for the learner to succeed but the top-down approach to imposing supplementary virtual learning classes on parents is the antithesis of this. Most parents will testify that nobody asked them what they needed and what would work for their children, schedules or homes.The notion of synchronous virtual learning, which most schools have suggested to or imposed on parents, is as utopian as it is unreasonable, given the limited computer access in homes as well as spotty WiFi, among other obstacles like the expenses involved and the crazy work schedules that parents who work in essential services have been forced to adapt to because of the pandemic. Kenya National Union of Teachers Secretary-General Wilson Sossion has pointed out that the practicability of virtual lessons is far from being a reality in Kenya, adding that supervision in a manner thatâ€™s useful to the students is also an issue. Despite much hype by the government and private schools about virtual learning, one of the things they fail to acknowledge is that it will leave out a significant proportion of learners. It will be survival for the wealthiest, not the fittest.Data from the Kenya National Bureau of Statistics (KNBS) revealed that the digital divide in Kenya is still a reality, as only one in five Kenyans have access to the internet. BLANKET SOLUTIONS The income of the parents also determines whether their children will access virtual lessons or not. The National ICT Survey Report (2018) by the Communication Authority and the KNBS indicated that employment and household size in many cases determine the household disposable income, which in turn determines whether individuals in households can afford radios, TVs, computers, internet and other ICT equipment. Whatâ€™s clear is that by offering blanket solutions for virtual learning, private schools are just combing a dry riverbed. There is an opportunity here to respond to the abnormal times abnormally, to paraphrase Health Cabinet Secretary Mutahi Kagweâ€™s now-famous quote about Covid-19. In their quest to keep afloat, private schools should not forget to fully involve parents in designing solutions that will accommodate different learnerâ€™s and parentsâ€™ needs. Nobody will walk away from this pandemic unscathed, but as everything comes unglued after this, private schools can help lessen the impact by doing right by parents and learners"
    # predict_class(text, class1_stats, class2_stats, class3_stats,
    #           class1_occurrence_ratio, class1_occurrence_ratio, class3_occurrence_ratio)
    return([class1_stats,class2_stats,class3_stats,class1_occurrence_ratio,class2_occurrence_ratio,class3_occurrence_ratio])


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


def test_classifier(class1_stats, class2_stats, class3_stats, class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio):
    dataset = load_dataset('test_data.csv')
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3}
    # test class 1
    # print('TESTING CLASS 1 - FAITH ONEYA')
    i = 0
    while i < 5:
        stopwords_frequency = stopwords_frequency_ratio(dataset[i][1])
        average_sentenceLength = average_sentence_length(dataset[i][1])
        simple_sentence_distribution = sentence_length_distribution(
            dataset[i][1])[0]
        compound_sentence_distribution = sentence_length_distribution(
            dataset[i][1])[1]
        complex_sentence_distribution = sentence_length_distribution(
            dataset[i][1])[2]

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
        # print(class1_probability, class2_probability, class3_probability)
        if(class1_probability > class2_probability and class1_probability > class3_probability):
            print('Predicted Class 1 - Faith Oneya')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga')
        i += 1
    # test class 2
    # print('TESTING CLASS 2 - BITANGE NDEMO')
    i = 0
    while i < 5:
        stopwords_frequency = stopwords_frequency_ratio(dataset[5+i][1])
        average_sentenceLength = average_sentence_length(dataset[5+i][1])
        simple_sentence_distribution = sentence_length_distribution(
            dataset[5+i][1])[0]
        compound_sentence_distribution = sentence_length_distribution(
            dataset[5+i][1])[1]
        complex_sentence_distribution = sentence_length_distribution(
            dataset[5+i][1])[2]

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
        # print(class1_probability, class2_probability, class3_probability)
        if(class1_probability > class2_probability and class1_probability > class3_probability):
            print('Predicted Class 1 - Faith Oneya')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga')
        i += 1
    # test class 3
    # print('TESTING CLASS 3 - ABIGAIL ARUNGA')
    i = 0
    while i < 5:
        stopwords_frequency = stopwords_frequency_ratio(dataset[10+i][1])
        average_sentenceLength = average_sentence_length(dataset[10+i][1])
        simple_sentence_distribution = sentence_length_distribution(
            dataset[10+i][1])[0]
        compound_sentence_distribution = sentence_length_distribution(
            dataset[10+i][1])[1]
        complex_sentence_distribution = sentence_length_distribution(
            dataset[10+i][1])[2]

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
        # print(class1_probability, class2_probability, class3_probability)
        if(class1_probability > class2_probability and class1_probability > class3_probability):
            print('Predicted Class 1 - Faith Oneya')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga')
        i += 1

@app.route('/author_id/api/predict_class', methods=['POST'])
def predict_class():
    req_data = request.get_json()
    text = req_data['raw_text']
    print(req_data['raw_text'])
    model=naive_bayes()
    class1_stats = model[0]
    class2_stats=model[1]
    class3_stats=model[2]
    class1_occurrence_ratio=model[3]
    class2_occurrence_ratio=model[4]
    class3_occurrence_ratio=model[5]

    # stopwords freq
    stopwords_frequency = stopwords_frequency_ratio(text)
    # average sentence length
    average_sentenceLength = average_sentence_length(text)
    # simple sentence distribution
    simple_sentence_distribution=sentence_length_distribution(text)[0]
    # compound sentence distribution
    compound_sentence_distribution=sentence_length_distribution(text)[1]
    # complex sentence distribution
    complex_sentence_distribution=sentence_length_distribution(text)[2]

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
    # print(class1_probability, class2_probability, class3_probability)
    predictedClass = ''
    if(class1_probability > class2_probability and class1_probability > class3_probability):
        print('Predicted Class 1 - Faith Oneya')
        predictedClass = 'Predicted Class 1 - Faith Oneya'
    elif(class2_probability > class1_probability and class2_probability > class3_probability):
        print('Predicted Class 2 - Bitange Ndemo')
        predictedClass = 'Predicted Class 2 - Bitange Ndemo'
    elif(class3_probability > class1_probability and class3_probability > class2_probability):
        print('Predicted Class 3 - Abigail Arunga')
        predictedClass = 'Predicted Class 3 - Abigail Arunga'
    # return predictedClass
    return jsonify({'predicted_class': predictedClass}), 200

    # print(sentence_length_distribution('When it’s not an integer, the highest probability number of events will be the nearest integer to the rate parameter, since the Poisson distribution is only defined for a discrete number of events. The discrete nature of the Poisson distribution is also why this is a probability mass function and not a density function. We can use the Poisson Distribution mass function to find the probability of observing a number of events over an interval generated by a Poisson process. Another use of the mass function equation — as we’ll see later — is to find the probability of waiting some time between events. For the problem we’ll solve with a Poisson distribution, we could continue with website failures, but I propose something grander. In my childhood, my father would often take me into our yard to observe (or try to observe) meteor showers. We were not space geeks, but watching objects from outer space burn up in the sky was enough to get us outside even though meteor showers always seemed to occur in the coldest months. The number of meteors seen can be modeled as a Poisson distribution because the meteors are independent, the average number of meteors per hour is constant (in the short term), and this is an approximation meteors don’t occur simultaneously. To characterize the Poisson distribution, all we need is the rate parameter which is the number of events/interval * interval length. From what I remember, we were told to expect 5 meteors per hour on average or 1 every 12 minutes. Due to the limited patience of a young child (especially on a freezing night), we never stayed out more than 60 minutes, so we’ll use that as the time period. Putting the two together, we get:'))
# naive_bayes()
# print(calculate_probability(1.0,1.0,1.0))
# ]
# predict_class('Faced with the daunting reality of empty accounts post-Covid-19, private schools have been forced to take control of their financial fate by reinventing their income streams at the expense of parents and learners. With all the attention focused on the war against Covid-19, most schools have taken the opportunity to make outrageous demands in terms of school fees, safe in the knowledge that their actions might barely be noticed. To be fair, the pandemic has forced some hard choices on schools, which were first to be closed on March 15 as they are considered high-risk areas because of the many learners they hold, and who come from different backgrounds. The Daily Nation recently carried a story showing how fancy schools are struggling to stay afloat. Nevertheless, this doesnâ€™t negate the fact that some of the decisions taken by schools â€” like rushing to charge them astronomical amounts for virtual learning â€” have harmed parents. When Brookhouse School demanded a 90 per cent school fees payment from parents for online learning, they fought back by suing the school and the court compelled the institution to reduce the fees by half. Kudos to the parents. While one canâ€™t extrapolate too much on the case of the Brookhouse School parents, given that they are not a towering example of the average Kenyan parent, itâ€™s still a valid example of the frustrations most parents are feeling with their childrenâ€™s schools.SCARCE RESOURCES The proponents of the Competency-Based Curriculum, to which a majority of schools in Kenya subscribe, boast about the level of involvement from a parent thatâ€™s needed for the learner to succeed but the top-down approach to imposing supplementary virtual learning classes on parents is the antithesis of this. Most parents will testify that nobody asked them what they needed and what would work for their children, schedules or homes.The notion of synchronous virtual learning, which most schools have suggested to or imposed on parents, is as utopian as it is unreasonable, given the limited computer access in homes as well as spotty WiFi, among other obstacles like the expenses involved and the crazy work schedules that parents who work in essential services have been forced to adapt to because of the pandemic. Kenya National Union of Teachers Secretary-General Wilson Sossion has pointed out that the practicability of virtual lessons is far from being a reality in Kenya, adding that supervision in a manner thatâ€™s useful to the students is also an issue. Despite much hype by the government and private schools about virtual learning, one of the things they fail to acknowledge is that it will leave out a significant proportion of learners. It will be survival for the wealthiest, not the fittest.Data from the Kenya National Bureau of Statistics (KNBS) revealed that the digital divide in Kenya is still a reality, as only one in five Kenyans have access to the internet. BLANKET SOLUTIONS The income of the parents also determines whether their children will access virtual lessons or not. The National ICT Survey Report (2018) by the Communication Authority and the KNBS indicated that employment and household size in many cases determine the household disposable income, which in turn determines whether individuals in households can afford radios, TVs, computers, internet and other ICT equipment. Whatâ€™s clear is that by offering blanket solutions for virtual learning, private schools are just combing a dry riverbed. There is an opportunity here to respond to the abnormal times abnormally, to paraphrase Health Cabinet Secretary Mutahi Kagweâ€™s now-famous quote about Covid-19. In their quest to keep afloat, private schools should not forget to fully involve parents in designing solutions that will accommodate different learnerâ€™s and parentsâ€™ needs. Nobody will walk away from this pandemic unscathed, but as everything comes unglued after this, private schools can help lessen the impact by doing right by parents and learners')
