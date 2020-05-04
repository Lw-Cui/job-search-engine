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

from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

docs = None


def init(dataset):
    global docs
    docs = read_docs(dataset)


def query(intro: str):
    query = [intro, "", "", "", "", "", "", "", ""]
    queries = generate_queries([query,])
    term_funcs = {
        # 'tf': compute_tf,
        'tfidf': compute_tfidf
        # 'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim
        # 'jaccard': jaccard_sim,
        # 'dice': dice_sim,
        # 'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [True],  # stem
        [True],  # remove stopwords
        sim_funcs,
        [TermWeights(company=1, title=1, category=1, location=1, description=1, mini_qual=1, pref_qual=1)]
    ]

    results = []
    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):

        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_num = len(processed_docs)
        doc_tf_vectors = [compute_tf(doc, doc_freqs, term_weights, doc_num) for doc in processed_docs]

        total_dl = 0.0
        for doc_tf_vec in doc_tf_vectors:
            total_dl += doc_tf_vec['_dl']

        avg_dl = total_dl / doc_num
        
        metrics = []

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


def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels


def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = []  # empty 0 index

    df = preprocess_data(file, tag='Amazon')
    return [Document(i + 1, row['Company'], row['Title'], row['Category'], row['Location'],
                     row['Responsibilities'], row['Minimum_Qualifications'], row['Preferred_Qualifications']) for i, row
            in df.iterrows()]


def preprocess_data(file, tag):
    if tag == 'Amazon':
        df = pd.read_csv(file)
        df.iloc[:, 0] = 'amazon'
        df.drop(['Time'], axis=1, inplace=True)
        df.insert(2, 'Category', df['Title'])

    if tag == 'Google':
        df = pd.read_csv(file)
        df = df.rename(columns={'Minimum Qualifications': 'Minimum_Qualifications',
                                'Preferred Qualifications': 'Preferred_Qualifications'})
    df = df.dropna(how='any', axis='rows')
    df['Company'] = df.Company.apply(lambda x: str(x).lower())
    df['Company'] = df.Company.apply(lambda x: word_tokenize(x))

    df['Title'] = df.Title.apply(lambda x: x.lower())
    df['Title'] = df.Title.apply(lambda x: word_tokenize(x))
    df['Category'] = df.Category.apply(lambda x: x.lower())
    df['Category'] = df.Category.apply(lambda x: word_tokenize(x))
    df['Location'] = df.Location.apply(lambda x: x.lower())
    df['Location'] = df.Location.apply(lambda x: word_tokenize(x))

    df['Responsibilities'] = df.Responsibilities.apply(lambda x: x.lower())
    df['Responsibilities'] = df.Responsibilities.apply(lambda x: word_tokenize(x))
    df['Responsibilities'] = df.Responsibilities.apply(lambda x: [w for w in x if w not in stop_words])

    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: x.lower())
    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: word_tokenize(x))
    df['Minimum_Qualifications'] = df.Minimum_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])

    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: x.lower())
    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: word_tokenize(x))
    df['Preferred_Qualifications'] = df.Preferred_Qualifications.apply(lambda x: [w for w in x if w not in stop_words])

    return df

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

# TODO: put any extensions here
def read_from_keyboard():
    flag = 'Yes'
    queries_text = []

    while flag == 'Yes':
        flag = input('Do you want to continue inputting query: (Yes/No)')
        if flag == 'No':
            break
        query = [[], [], [], [], [], [], []]
        query[0] = input('Company: ')
        query[1] = input('Job Title: ')
        query[2] = input('Job Category: ')
        query[3] = input('Job Location: ')
        query[4] = input('Job Description: ')
        query[5] = input('Minimum Qualification: ')
        query[6] = input('Preferred Qualification: ')

        queries_text += [query]

    return generate_queries(queries_text)


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


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    processed_queries = remove_stopwords(processed_queries)

    # TODO: Add stem parameter
    processed_docs = stem_docs(processed_docs, stem)
    processed_queries = stem_docs(processed_queries, stem)

    return processed_docs, processed_queries

def search_debug(doc_tf_vectors, avg_dl, doc_freqs, N, query):
    results_with_score = [(doc_id + 1, compute_bm25f_score_fulltext(doc_tf_vec, avg_dl, doc_freqs, N, query))
                          for doc_id, doc_tf_vec in enumerate(doc_tf_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    output = []
    print('Query:', query)
    print()
    for doc_id, score in results_with_score[:10]:
        output.append(docs[doc_id - 1])
    return output
