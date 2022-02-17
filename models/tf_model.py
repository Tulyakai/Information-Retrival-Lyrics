from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import dataProcess as d


if __name__ == '__main__':
    parsed_data = pickle.load(open('../assets/parsed_data.pkl', 'rb'))
    vectorizer = CountVectorizer(ngram_range=(1, 3), preprocessor=d.preProcess)
    X = vectorizer.fit_transform(parsed_data['lyric'])
    X.data = np.log10(X.data + 1)

    pickle.dump((vectorizer, X) , open('../models/tf.pkl' ,'wb'))

