# Information-Retrieval-Lyrics
This is a midterm project of 622115012 - 953481 Information Retrieval course, CAMT, CMU.

### Technology Tools
- Python
- Flask
### IR technique
- TF
- TF-IDF
- BM25
# Setting guide
1. Dowload required lyric dataset from [here](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres), then paste into assets folder.
2. Dowload required wiki datasets(100K 2020, 300K, 100K 2016, 1M 2016) from [here](https://wortschatz.uni-leipzig.de/en/download/English?fbclid=IwAR3bjZtPuJiAdXus-oPEImcU7E0ErzH7onI2ih4cAVjMPisFOBFBlYitQno), then paste into assets folder.
3. Run main in dataProcess.py to create parsed_data.pkl and clean_wiki_100.txt.
4. Run main in bm25_model.py, tf_model.py, and tfidf_model.py to create fitted mdoel and vectorizer.
5. Run main in main.py, then go to http://localhost:5000/index.
6. Enjoy Searching 🥣
