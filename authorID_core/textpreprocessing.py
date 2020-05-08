# Function to remove stopwords from a tweet array.
# Stop words are very common words that can be removed without changing the meaning of a text.
def remove_stopwords(tweet_array):
    import ntlk
    ntlk.download('stopwords')
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words('english')
    processed_tweet_array = []
    for tweet in tweet_array:
        processed_tweet_array.append(
            ' '.join([word for word in tweet.split()
                      if word not in english_stop_words])
        )
    return processed_tweet_array

# Function to reduce words to their roots(eng language).
# This reduces all words that are inflections of the same word to their root so they are the same
# Playing, Play, Played => play
# this step may produce words that are not actual words
def stemmer(tweet_array):
    from nltk.stem import PorterStemmer
    porter = PorterStemmer()
    stemmed_tweet_array = []
    for tweet in tweet_array:
        stemmed_tweet_array.append(
            ' '.join(porter.stem(word)
                     for word in tweet.split()))
    return stemmed_tweet_array


print(stemmer(remove_stopwords(['I am playing a game', 'I work at ISW',
                                'I am a developer', 'I should be working'])))
