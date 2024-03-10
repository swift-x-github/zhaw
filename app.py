from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import scipy
import os
import pandas as pd

model = SentenceTransformer('bert-base-nli-mean-tokens')

# setting up template directory
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)

@app.route("/")
def hello():
	return TEMPLATE_DIR

# A corpus is a list with documents split by sentences.
BASE_DIR = './'
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'archive')
NEWS_FILE_NAME = "abcnews-date-text.csv"

def read_csv(filepath):
     if os.path.splitext(filepath)[1] != '.csv':
          return  # or whatever
     seps = [',', ';', '\t']                    # ',' is default
     encodings = [None, 'utf-8', 'ISO-8859-1']  # None is default
     for sep in seps:
         for encoding in encodings:
              try:
                  return pd.read_csv(filepath, encoding=encoding, sep=sep)
              except Exception:  # should really be more specific 
                  pass
     raise ValueError("{!r} is has no encoding in {} or seperator in {}"
                      .format(filepath, encodings, seps))


input_df = read_csv(os.path.join(TEXT_DATA_DIR, NEWS_FILE_NAME))
input_df = input_df.head(2000)
print(input_df.head(20))

sentences = input_df['headline_text'].values.tolist()

# Each sentence is encoded as a 1-D vector with 78 columns
print("Getting embeddings for sentences ....")
sentence_embeddings = model.encode(sentences)
print("done with getting embeddings for sentences ....")

print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))

def performSearch(query):
	queries = [query]
	query_embeddings = model.encode(queries)

	# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
	number_top_matches = 3 #@param {type: "number"}

	print("Semantic Search Results")
	results = []
	for query, query_embedding in zip(queries, query_embeddings):
		distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])
    	
	return results

@app.route("/semanticsearch",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		query = request.form.get('query')
		print(query)
		results = performSearch(query)
		return render_template('semantic_search.html', query=query, results=results, sentences=sentences)
	else:
		return render_template('semantic_search.html', review="" ,results=None)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9200, debug=True, threaded=True)
