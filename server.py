from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
api = Api(app)


class cosine_sim(Resource):
    def get(self, sent1, sent2):
        stemmer = nltk.stem.porter.PorterStemmer()
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]

        '''remove punctuation, lowercase, stem'''
        def normalize(text):
            return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

        vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

        #return cosine similarity
        tfidf = vectorizer.fit_transform([sent1, sent2])
        result = { 'cosine_sim' : ((tfidf * tfidf.T).A)[0,1]}
        return jsonify(result)


api.add_resource(cosine_sim, '/<sent1>/<sent2>')


if __name__ == '__main__':
     app.run(port='5002')