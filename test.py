# -*- coding: utf-8 -*-
import nltk
# import statistics
import scipy.stats
import math
from math import sqrt
from math import pi
from math import exp
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
            stopwords_frequency_ratio(dataset[10+i][1]))
        class3_stopwords_frequency_arr.append(
            stopwords_frequency_ratio(dataset[20+i][1]))
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


# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# calculate standard deviation
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

def test_classifier(class1_stats, class2_stats, class3_stats, dataset, class1_occurrence_ratio, class2_occurrence_ratio, class3_occurrence_ratio):
    labels = {'Faith Oneya': 1, 'Bitange Ndemo': 2, 'Abigail Arunga': 3}
    # test class 1
    print('TESTING CLASS 1 - FAITH ONEYA')
    i = 0
    while i < 10:
        stopwords_frequency = stopwords_frequency_ratio(dataset[i][1])
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
            print('Predicted Class 1 - Faith Oneya, Actual - Faith Oneya')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo, Actual - Faith Oneya')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga, Actual - Faith Oneya')
        i += 1
    # test class 2
    print('TESTING CLASS 2 - BITANGE NDEMO')
    i = 0
    while i < 10:
        stopwords_frequency = stopwords_frequency_ratio(dataset[10+i][1])
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
            print('Predicted Class 1 - Faith Oneya, Actual - Bitange Ndemo')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo, Actual - Bitange Ndemo')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga, Actual - Bitange Ndemo')
        i += 1
    # test class 3
    print('TESTING CLASS 3 - ABIGAIL ARUNGA')
    i = 0
    while i < 10:
        stopwords_frequency = stopwords_frequency_ratio(dataset[20+i][1])
        simple_sentence_distribution = sentence_length_distribution(
            dataset[20+i][1])[0]
        compound_sentence_distribution = sentence_length_distribution(
            dataset[20+i][1])[1]
        complex_sentence_distribution = sentence_length_distribution(
            dataset[20+i][1])[2]

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
            print('Predicted Class 1 - Faith Oneya, Actual - Abigail Arunga')
        elif(class2_probability > class1_probability and class2_probability > class3_probability):
            print('Predicted Class 2 - Bitange Ndemo, Actual - Abigail Arunga')
        elif(class3_probability > class1_probability and class3_probability > class2_probability):
            print('Predicted Class 3 - Abigail Arunga, Actual - Abigail Arunga')
        i += 1


    # print(sentence_length_distribution('When it’s not an integer, the highest probability number of events will be the nearest integer to the rate parameter, since the Poisson distribution is only defined for a discrete number of events. The discrete nature of the Poisson distribution is also why this is a probability mass function and not a density function. We can use the Poisson Distribution mass function to find the probability of observing a number of events over an interval generated by a Poisson process. Another use of the mass function equation — as we’ll see later — is to find the probability of waiting some time between events. For the problem we’ll solve with a Poisson distribution, we could continue with website failures, but I propose something grander. In my childhood, my father would often take me into our yard to observe (or try to observe) meteor showers. We were not space geeks, but watching objects from outer space burn up in the sky was enough to get us outside even though meteor showers always seemed to occur in the coldest months. The number of meteors seen can be modeled as a Poisson distribution because the meteors are independent, the average number of meteors per hour is constant (in the short term), and this is an approximation meteors don’t occur simultaneously. To characterize the Poisson distribution, all we need is the rate parameter which is the number of events/interval * interval length. From what I remember, we were told to expect 5 meteors per hour on average or 1 every 12 minutes. Due to the limited patience of a young child (especially on a freezing night), we never stayed out more than 60 minutes, so we’ll use that as the time period. Putting the two together, we get:'))
naive_bayes()
# print(calculate_probability(1.0,1.0,1.0))
