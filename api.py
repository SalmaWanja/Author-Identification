import flask
import flask_cors
from flask import request, jsonify, Flask
from flask_cors import CORS

# import nltk
# nltk.download('stopwords') 
app = Flask(__name__)
CORS(app)


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

# Classifier. 
class KNN_NLC_Classifer():
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type

    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match. 
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            max_sim = 0
            max_index = 0
            for j in range(self.x_train.shape[0]):
                temp = self.document_similarity(x_test[i], self.x_train[j])
                if temp > max_sim:
                    max_sim = temp
                    max_index = j
            y_predict.append(self.y_train[max_index])
        return y_predict


# conversion of document to syn sets
def convert_tag(self, tag):
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None


    def doc_to_synsets(self, doc):
        """
            Returns a list of synsets in document.
            Tokenizes and tags the words in the document doc.
            Then finds the first synset for each word/tag combination.
        If a synset is not found for that combination it is skipped.

        Args:
            doc: string to be converted

        Returns:
            list of synsets
        """
        tokens = word_tokenize(doc+' ')
        
        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)
        
        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l  


# PROVIDES THE SIMILARITY SCORE BETWEEN TWO TEXTS USING THEIR SYNSETS
def similarity_score(self, s1, s2, distance_type = 'path'):
          """
          Calculate the normalized similarity score of s1 onto s2
          For each synset in s1, finds the synset in s2 with the largest similarity value.
          Sum of all of the largest similarity values and normalize this value by dividing it by the
          number of largest similarity values found.

          Args:
              s1, s2: list of synsets from doc_to_synsets

          Returns:
              normalized similarity score of s1 onto s2
          """
          s1_largest_scores = []

          for i, s1_synset in enumerate(s1, 0):
              max_score = 0
              for s2_synset in s2:
                  if distance_type == 'path':
                      score = s1_synset.path_similarity(s2_synset, simulate_root = False)
                  else:
                      score = s1_synset.wup_similarity(s2_synset)                  
                    if score != None:
                      if score > max_score:
                          max_score = score
              
              if max_score != 0:
                  s1_largest_scores.append(max_score)
          
          mean_score = np.mean(s1_largest_scores)
                 
          return mean_score  