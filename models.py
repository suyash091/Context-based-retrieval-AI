import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFmodel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values, data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]


def random_model(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)