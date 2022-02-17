from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import dataProcess as d

if __name__ == '__main__':
    parsed_data = pickle.load(open('../assets/parsed_data.pkl', 'rb'))
    vectorizer = TfidfVectorizer(ngram_range=(1,3), preprocessor=d.preProcess)
    X = vectorizer.fit_transform(parsed_data['lyric'])

    pickle.dump((vectorizer, X) , open('../models/tfidf.pkl' ,'wb'))

