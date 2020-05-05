import itertools
import re
import math
import pandas as pd
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

from io import StringIO
import numpy as np
import nltk
nltk.data.path.append("./nltk_data")

from ast import literal_eval

from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

docs = None
original_docs = None
doc_freqs = None
doc_num = 0
doc_tf_vectors = None
avg_dl = 0.0


def init(dataset):
    global docs
    global original_docs
    global doc_freqs
    global doc_num
    global doc_tf_vectors
    global avg_dl
    docs = read_docs(dataset)
    original_docs = docs

    processed_docs = process_docs(docs, True, False, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_num = len(processed_docs)
    doc_tf_vectors = [compute_tf(doc, doc_freqs, TermWeights(company=1, title=1, category=1, location=1, description=1, mini_qual=1, pref_qual=1), doc_num) for doc in processed_docs]

    total_dl = 0.0
    for doc_tf_vec in doc_tf_vectors:
        total_dl += doc_tf_vec['_dl']

    avg_dl = total_dl / doc_num


def query(intro: str):
    global docs
    global original_docs
    global doc_freqs
    global doc_num
    global doc_tf_vectors
    global avg_dl
    query = [intro, "", "", "", "", "", "", "", ""]
    queries = generate_queries([query,])
    processed_queries = process_docs(queries, True, True, stopwords)
    results = []
    for query in processed_queries:
        results = search_debug(doc_tf_vectors, avg_dl, doc_freqs, doc_num, query)
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
                     row['Responsibilities'], row['Minimum_Qualifications'], row['Preferred_Qualifications']) for i, row
            in df.iterrows()]

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

### Term-Document Matrix

class TermWeights(NamedTuple):
    company: float
    title: float
    category: float
    location: float
    description: float
    mini_qual: float
    pref_qual: float


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


def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list, N=0):
    vec = {
        'company': defaultdict(float),
        'title': defaultdict(float),
        'category': defaultdict(float),
        'location': defaultdict(float),
        'description': defaultdict(float),
        'mini_qual': defaultdict(float),
        'pref_qual': defaultdict(float),
        '_full': defaultdict(float),
        '_dl': 0.0
    }
    for word in doc.company:
        vec['company'][word] += weights.company
        vec['_full'][word] += weights.company
        vec['_dl'] += weights.company
    for word in doc.title:
        vec['title'][word] += weights.title
        vec['_full'][word] += weights.title
        vec['_dl'] += weights.title
    for word in doc.category:
        vec['category'][word] += weights.category
        vec['_full'][word] += weights.category
        vec['_dl'] += weights.category
    for word in doc.location:
        vec['location'][word] += weights.location
        vec['_full'][word] += weights.location
        vec['_dl'] += weights.location
    for word in doc.description:
        vec['description'][word] += weights.description
        vec['_full'][word] += weights.description
        vec['_dl'] += weights.description
    for word in doc.mini_qual:
        vec['mini_qual'][word] += weights.mini_qual
        vec['_full'][word] += weights.mini_qual
        vec['_dl'] += weights.mini_qual
    for word in doc.pref_qual:
        vec['pref_qual'][word] += weights.pref_qual
        vec['_full'][word] += weights.pref_qual
        vec['_dl'] += weights.pref_qual

    # convert back to a regular dict
    vec = {
        'company': dict(vec['company']),
        'title': dict(vec['title']),
        'category': dict(vec['category']),
        'location': dict(vec['location']),
        'description': dict(vec['description']),
        'mini_qual': dict(vec['mini_qual']),
        'pref_qual': dict(vec['pref_qual']),
        '_full': dict(vec['_full']),
        '_dl': vec['_dl']
    }
    return vec


def compute_tfidf(doc, doc_freqs, weights, N):
    vec_dict = compute_tf(doc, doc_freqs, weights)
    tfidf_vec = {}

    for word in vec_dict.keys():
        if word not in doc_freqs.keys():
            continue
        else:
            tfidf_vec[word] = vec_dict[word] * math.log(N / doc_freqs[word])
    return tfidf_vec

def compute_bm25f_score(doc_tf_vec, avg_dl, doc_freqs, N, query):
    k1 = 1.2
    b = 0.75
    len_norm_component = 1 - b + b * (doc_tf_vec['_dl'] / avg_dl)

    query_vec = defaultdict(float)
    for word in query.company:
        if word in doc_tf_vec['company'].keys():
            query_vec[word] += doc_tf_vec['company'][word]
    for word in query.title:
        if word in doc_tf_vec['title'].keys():
            query_vec[word] += doc_tf_vec['title'][word]
    for word in query.category:
        if word in doc_tf_vec['category'].keys():
            query_vec[word] += doc_tf_vec['category'][word]
    for word in query.location:
        if word in doc_tf_vec['location'].keys():
            query_vec[word] += doc_tf_vec['location'][word]
    for word in query.description:
        if word in doc_tf_vec['description'].keys():
            query_vec[word] += doc_tf_vec['description'][word]
    for word in query.mini_qual:
        if word in doc_tf_vec['mini_qual'].keys():
            query_vec[word] += doc_tf_vec['mini_qual'][word]
    for word in query.pref_qual:
        if word in doc_tf_vec['pref_qual'].keys():
            query_vec[word] += doc_tf_vec['pref_qual'][word]

    rsv = 0.0
    for word in query_vec.keys():
        if word in doc_freqs.keys():
            rsv += math.log(N / doc_freqs[word]) * (k1 + 1) * query_vec[word] / (k1 * len_norm_component + query_vec[word])

    return rsv

def compute_bm25f_score_fulltext(doc_tf_vec, avg_dl, doc_freqs, N, query):
    k1 = 1.2
    b = 0.75
    len_norm_component = 1 - b + b * (doc_tf_vec['_dl'] / avg_dl)

    query_vec = defaultdict(float)
    for sec in query.sections():
        for word in sec:
            if word in doc_tf_vec['_full'].keys():
                query_vec[word] += doc_tf_vec['_full'][word]
    rsv = 0.0
    for word in query_vec.keys():
        if word in doc_freqs.keys():
            rsv += math.log(N / doc_freqs[word]) * (k1 + 1) * query_vec[word] / (k1 * len_norm_component + query_vec[word])

    return rsv

### Vector Similarity

def cosine_sim(x, y):
    return 0

def split_query_text(content):
    res = []
    for word in word_tokenize(content):
        res += [word.lower()]
    return res


def generate_queries(queries_text):
    queries = []
    for i, query in enumerate(queries_text):
        company = split_query_text(query[0].lower())
        title = split_query_text(query[1].lower())
        category = split_query_text(query[2].lower())
        location = split_query_text(query[3].lower())
        description = split_query_text(query[4].lower())
        mini_qual = split_query_text(query[5].lower())
        pref_qual = split_query_text(query[6].lower())

        queries += [Document(i + 1, company, title, category, location, description, mini_qual, pref_qual)]

    return queries


def process_docs(docs, stem, removestop, stopwords):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)

    # TODO: Add stem parameter
    if stem:
        processed_docs = stem_docs(processed_docs, stem)

    return processed_docs

def search_debug(doc_tf_vectors, avg_dl, doc_freqs, N, query):
    results_with_score = [(doc_id + 1, compute_bm25f_score_fulltext(doc_tf_vec, avg_dl, doc_freqs, N, query))
                          for doc_id, doc_tf_vec in enumerate(doc_tf_vectors)]
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

