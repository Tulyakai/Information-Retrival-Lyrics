from flask import Flask, request, render_template
from flask_cors import CORS
import pandas as pd
import string
import pickle
import re
from models.bm25_model import BM25
from spellchecker import SpellChecker

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

parsed_data = pickle.load(open('assets/parsed_data.pkl', 'rb'))
#tf
tf_vec, tf_X = pickle.load(open('models/tf.pkl', 'rb'))
#tfidf
tfidf_vec, tfidf_X = pickle.load(open('models/tfidf.pkl', 'rb'))
#bm25
bm25 = pickle.load(open('models/bm25.pkl', 'rb'))


#spell correction
spell = SpellChecker(language='en')
spell.word_frequency.load_text_file('assets/clean_wiki_100k.txt')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/<page>')
def hello(page):
    X = int(page)
    if X > 1:
        return {'songs':parsed_data[(X-1)*10:X*10].to_dict('records'), 'page':X}, 200
    else:
        return {'songs':parsed_data[:10].to_dict('records'), 'page':1}, 200

@app.route('/query-lyric', methods=['POST'])
def queryLyric():
    body = request.get_json()
    if (body['query'] is None) or (body['score'] is None):
        return  {'error': 'Request body must contain query and score'}, 400
    else:
        body['score'] = body['score'].replace(' ', '')
        body['score'] = body['score'].lower().translate(str.maketrans('', '', string.punctuation))
        if(body['score'] == 'tf'):
            query_vec = tf_vec.transform([body['query']])
            cos_sim = cosine_similarity(tf_X, query_vec).reshape((-1), )
            df_tf = pd.DataFrame({'tf': list(cos_sim), 'lyric': list(parsed_data['lyric']), 'song':list(parsed_data['song']), 'artist':list(parsed_data['artist'])}).nlargest(columns='tf', n=10)
            df_tf['rank'] = df_tf['tf'].rank(ascending=False)
            df_tf = df_tf.drop(columns='tf', axis=1)
            spell_corr = [spell.correction(w) for w in body['query'].split()]
            return {'songs': df_tf.to_dict('records'), 'candidate_query': ' '.join(spell_corr)}

        elif(body['score'] == 'tfidf'):
            query_vec = tfidf_vec.transform([body['query']])
            cos_sim = cosine_similarity(tfidf_X, query_vec).reshape((-1), )
            df_tf_idf = pd.DataFrame({'tfidf': list(cos_sim), 'lyric': list(parsed_data['lyric']), 'song':list(parsed_data['song']), 'artist':list(parsed_data['artist'])}).nlargest(columns='tfidf', n=10)
            df_tf_idf['rank'] = df_tf_idf['tfidf'].rank(ascending=False)
            df_tf_idf = df_tf_idf.drop(columns='tfidf', axis=1)
            spell_corr = [spell.correction(w) for w in body['query'].split()]
            return {'songs': df_tf_idf.to_dict('records'), 'candidate_query': ' '.join(spell_corr)}
        elif(body['score'] == 'bm25'):
            score = bm25.transform(body['query'])
            df_bm = pd.DataFrame({'bm25': list(score), 'lyric': list(parsed_data['lyric']), 'song': list(parsed_data['song']), 'artist': list(parsed_data['artist'])}).nlargest(columns='bm25', n=10)
            df_bm['rank'] = df_bm['bm25'].rank(ascending=False)
            df_bm = df_bm.drop(columns='bm25', axis=1)
            spell_corr = [spell.correction(w) for w in body['query'].split()]
            return {'songs': df_bm.to_dict('records'), 'candidate_query': ' '.join(spell_corr)}
        else:
            return {'error': body['score'] + ' scoring in not exist'}, 400


@app.route('/query-artist', methods=['POST'])
def queryArtist():
    body = request.get_json()
    if (body['artist'] is None):
        return {'error': 'Request body must contain query and score'}, 400
    result = parsed_data[parsed_data['artist'] == body['artist']]
    result = result.sort_values('song')
    return {'artist': result.to_dict('records')}

@app.route('/query-song', methods=['POST'])
def querySong():
    body = request.get_json()
    if (body['song'] is None):
        return {'error': 'Request body must contain query and score'}, 400
    result = parsed_data[parsed_data['song'].str.contains(body['song'].lower())]

    result = result[result['song'].apply(lambda s: s.split()[0] == body['song'].lower().split()[0])]
    result = result[result['song'].apply(lambda s: len((re.sub(r'[\(\[].*?[\)\]]', '', s)).split()) ==  len(body['song'].split()))]

    return {'artist': result.to_dict('records')}

if __name__ == '__main__':
    app.run(debug=True)
