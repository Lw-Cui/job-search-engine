import itertools
import re
import math
import pandas as pd
import argparse
from ast import literal_eval
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

from io import StringIO
import numpy as np
import nltk
nltk.data.path.append("./nltk_data")

from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

docs = None
original_docs = None
glove_embeds = {}

def init(dataset):
    global docs
    global original_docs
    global glove_embeds
    docs = read_docs(dataset)
    original_docs = docs
    read_glove_embeddings('./data/glove.6B.50d.txt')
    print(glove_embeds['amazon'])

def query(intro: str):
    queries = generate_queries([intro])


    processed_docs, processed_queries = process_docs_and_queries(docs, queries, False, True, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_num = len(processed_docs)
    term_weights = TermWeights(company=1, title=1, category=1, location=1, description=1, mini_qual=1, pref_qual=1)
    doc_vectors = [generate_from_glove_doc(doc, doc_num, doc_freqs, term_weights) for doc in processed_docs]
    results = [] 
    for query in processed_queries:
        query_vec = generate_from_glove_query(query, doc_num, doc_freqs)
        results = search_debug(processed_docs, query, doc_vectors, query_vec, cosine_sim)
    return results


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    company: List[str]
    title: List[str]
    category: List[str]
    location: List[str]
    description: List[str]
    mini_qual: List[str]
    pref_qual: List[str]
    
    def sections(self):
        return [self.company, self.title, self.category, self.location, self.description, self.mini_qual,
                self.pref_qual]
    
    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
                f"  company: {self.company}\n" +
                f"  title: {self.title}\n" +
                f"  category: {self.category}\n" +
                f"  location: {self.location}\n" +
                f"  description: {self.description}\n" +
                f"  minimum qualification: {self.mini_qual}\n" +
                f"  preferred qualification: {self.pref_qual}\n")


stemmer = SnowballStemmer('english')

def read_docs(file):
    '''
        Reads the corpus into a list of Documents
        '''
    docs = []  # empty 0 index
    df = pd.read_csv(file)
    for col in df.columns:
        df[col] = df[col].apply(literal_eval)
    return [Document(i + 1, row['Company'], row['Title'], row['Category'], row['Location'],
                     row['Responsibilities'], row['Minimum_Qualifications'], row['Preferred_Qualifications']) for i, row in df.iterrows()]

def read_glove_embeddings(file):
    with open(file, 'r', encoding='utf—8') as f:
        for line in f.readlines():
            try:
                line = list(line.split())
                glove_embeds[line[0]] = np.array([float(num) for num in line[1:]])
            except:
                continue

def stem_doc(doc: Document, stem):
    new_doc = Document(doc.doc_id, [], [], [], [], [], [], [])
    if stem:
        new_doc = Document(doc.doc_id, *[[stemmer.stem(word) for word in sec] for sec in doc.sections()])
    else:
        new_doc = doc
    return new_doc


def stem_docs(docs: List[Document], stem):
    return [stem_doc(doc, stem) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stop_words]
                                  for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]

def compute_doc_freqs(docs: List[Document]):
    '''
        Computes document frequency, i.e. how many documents contain a specific word
        '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

### Term-Document Matrix

class TermWeights(NamedTuple):
    company: float
    title: float
    category: float
    location: float
    description: float
    mini_qual: float
    pref_qual: float

def compute_tf(doc, query):
    vec = defaultdict(float)
    if query:
        for word in doc:
            vec[word] += 1
    else:
        for sec in doc.sections():
            for word in sec:
                vec[word] += 1

    return dict(vec)

### Calculate TF-IDF
def compute_tfidf(doc, doc_freqs, N, query=False):
    vec_dict = compute_tf(doc, query)
    tfidf_vec = {}
    
    for word in vec_dict.keys():
        if word not in doc_freqs.keys():
            continue
        else:
            tfidf_vec[word] = vec_dict[word] * math.log(N / doc_freqs[word])
    return tfidf_vec

### Generate glove word embedding
def generate_from_glove_doc(doc, doc_num, doc_freqs, weights):
    tf_idf = compute_tfidf(doc, doc_freqs, doc_num)
    vec = np.zeros(50)
    cnt = len(doc.company) + len(doc.title) + len(doc.category) + len(doc.location) + len(doc.description) + len(doc.mini_qual) + len(doc.pref_qual)
    for word in doc.company:
        vec += weights.company * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.title:
        vec += weights.title * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.category:
        vec += weights.category * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.location:
        vec += weights.location * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.description:
        vec += weights.description * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.mini_qual:
        vec += weights.mini_qual * glove_embeds.get(word, 0) * tf_idf[word]
    for word in doc.pref_qual:
        vec += weights.pref_qual * glove_embeds.get(word, 0) * tf_idf[word]
    
    return vec / cnt

def generate_from_glove_query(query, doc_num, doc_freqs):
    tf_idf = compute_tfidf(query, doc_freqs, doc_num, query=True)
    vec = np.zeros(50)
    
    for word in query:
        vec += glove_embeds.get(word, 0) * tf_idf.get(word, 0)
    
    return vec / len(query)

### Vector Similarity

def cosine_sim(x, y):
    '''
        Computes the cosine similarity between two sparse term vectors represented as dictionaries.
        '''
    num = np.sum([a * b for a, b in zip(x, y)])
    return num / (norm(x) * norm(y))

### Extensions
def split_query_text(content):
    res = []
    for word in word_tokenize(content):
        res += [word.lower()]
    return res

def generate_queries(queries_text):
    queries = []
    for query_text in queries_text:
        queries += [split_query_text(query_text)]
    
    return queries

def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    
    processed_queries = queries
    if removestop:
        processed_queries = [[word for word in query if word not in stop_words] for query in processed_queries]
    if stem:
        processed_docs = stem_docs(processed_docs, stem)
        processed_queries = [[stemmer.stem(word) for word in query] for query in processed_queries]
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                          for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(_, query, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                          for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
                          
    output = []
    print('Query:', query)
    print()
    for doc_id, score in results_with_score[:10]:
        doc = original_docs[doc_id - 1]
        output.append({
            'company': ' '.join(doc.company),
            'title': ' '.join(doc.title),
            'category': ' '.join(doc.category),
            'location': ' '.join(doc.location),
            'description': ' '.join(doc.description),
            'minimum_qualifications': ' '.join(doc.mini_qual),
            'preferred_qualifications': ' '.join(doc.pref_qual),
        })
    return output

