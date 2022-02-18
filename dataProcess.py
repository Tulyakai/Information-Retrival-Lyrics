import fnmatch
from zipfile import ZipFile
import pandas as pd
import string
import re
import pickle
from nltk import word_tokenize, PorterStemmer

def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    # stopwords_set = set(stopwords.words())
    # stop_dict = {s: 1 for s in stopwords_set}
    # s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    s = s.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
    return s

def get_and_clean_data():
    # Lyrics DataFrame
    df_lyrics = pd.read_csv('assets/lyrics-data.csv')
    df_lyrics = df_lyrics[df_lyrics['Idiom'] == 'ENGLISH']
    df_lyrics = df_lyrics.drop_duplicates(subset='SLink')
    # Artist DataFrame
    df_artists = pd.read_csv('assets/artists-data.csv')
    df_artists = df_artists.drop_duplicates(subset='Link')
    # Merge
    df = pd.DataFrame.merge(df_lyrics, df_artists, how='inner', left_on='ALink', right_on='Link')
    # Drop columns and duplicate(['Artist', 'SName', 'Lyric'])
    df = df.drop(columns=['ALink', 'SLink', 'Idiom', 'Songs', 'Popularity', 'Link', 'Genre', 'Genres'], axis=1)
    df = df.drop_duplicates(subset=['Artist', 'SName', 'Lyric'])
    df = df.rename({'SName': 'song', 'Lyric': 'lyric', 'Artist': 'artist'}, axis='columns')
    df = df[['artist', 'song', 'lyric']]

    # clean lyric
    cleaned_lyric = df['lyric']
    cleaned_lyric = cleaned_lyric.apply(lambda s: re.sub(r'[\(\[].*?[\)\]]', '', s.lower()))
    cleaned_lyric = cleaned_lyric.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_lyric = cleaned_lyric.apply(lambda s: re.sub("\s+", " ", s.strip()))
    cleaned_lyric = cleaned_lyric.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    df['lyric'] = cleaned_lyric
    df['artist'] = df['artist'].apply(lambda s: s.lower())
    df['song'] = df['song'].apply(lambda s: s.lower())
    pickle.dump(df, open('assets/parsed_data.pkl' ,'wb'))
    return df


def clean_data_wiki_100k():
    f = open("assets/eng-simple_wikipedia_2021_100K-sentences.txt", "r", encoding='utf8')
    text = f.read()
    text = re.sub('[^A-za-z]', ' ', text.lower() )
    text = re.sub('\s+', ',', text)
    text = text.split(',')
    text = [w for w in text if len(w)>1]
    text = ' '.join(text)
    # save_text = open('assets/clean_wiki_100k.txt', 'r')
    # save_text.write(text)
    return text

def clean_data_wiki_100k_2016():
    f = open("assets/eng_wikipedia_2016_100K-sentences.txt", "r", encoding='utf8')
    text = f.read()
    text = re.sub('[^A-za-z]', ' ', text.lower() )
    text = re.sub('\s+', ',', text)
    text = text.split(',')
    text = [w for w in text if len(w)>1]
    text = ' '.join(text)
    # save_text = open('assets/clean_wiki_100k.txt', 'r')
    # save_text.write(text)
    return text

def clean_data_wiki_300k():
    f = open("assets/eng-simple_wikipedia_2021_300K-sentences.txt", "r", encoding='utf8')
    text = f.read()
    text = re.sub('[^A-za-z]', ' ', text.lower() )
    text = re.sub('\s+', ',', text)
    text = text.split(',')
    text = [w for w in text if len(w)>1]
    text = ' '.join(text)
    # save_text = open('assets/clean_wiki_300k.txt', 'w')
    # save_text.write(text)
    return text

def clean_data_wiki_1M():
    f = open("assets/eng_wikipedia_2016_1M-sentences.txt", "r", encoding='utf8')
    text = f.read()
    text = re.sub('[^A-za-z]', ' ', text.lower())
    text = re.sub('\s+', ',', text)
    text = text.split(',')
    text = [w for w in text if len(w)>1]
    text = ' '.join(text)
    # save_text = open('assets/clean_wiki_300k.txt', 'w')
    # save_text.write(text)
    return text

# def clean_iula():
#     with ZipFile('assets/IULA_Spanish-English_Technical_Corpus_data.zip') as zipfiles:
#         files = fnmatch.filter(zipfiles.namelist(), "EN/*/*plain.txt")
#         raw_IULA = [zipfiles.open(file_name).read().decode('utf8') for file_name in files]
#     text = ' '.join(raw_IULA)
#     text = re.sub('[^A-za-z]', ' ', text.lower())
#     text = re.sub('\s+', ',', text)
#     text = text.split(',')
#     text = [w for w in text if len(w) > 1]
#     text = ' '.join(text)
#     # save_text = open('assets/clean_wiki_300k.txt', 'w')
#     # save_text.write(text)
#     return text


def group_wiki():
    wiki_1 = clean_data_wiki_100k()
    wiki_2 = clean_data_wiki_300k()
    wiki_3 = clean_data_wiki_100k_2016()
    wiki_4 = clean_data_wiki_1M()
    # iula = clean_iula()
    wiki_1 += ' ' + wiki_2
    wiki_1 += ' ' + wiki_3
    wiki_1 += ' ' + wiki_4
    # wiki_1 += ' ' + iula
    save_text = open('assets/clean_wiki.txt', 'w')
    save_text.write(wiki_1)


if __name__ == '__main__':
    get_and_clean_data()
    group_wiki()
