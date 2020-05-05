
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
    queries = generate_queries([intro])
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
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, doc_num) for doc in processed_docs]
        
        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, doc_num, query=True)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            results = search_debug(processed_docs, query, doc_vectors, query_vec, sim_funcs[sim])
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


def compute_tf(doc: Document, weights: list):
    vec = defaultdict(float)
    for word in doc.company:
        vec[word] += weights.company
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.category:
        vec[word] += weights.category
    
    for word in doc.location:
        vec[word] += weights.location
    for word in doc.description:
        vec[word] += weights.description
    for word in doc.mini_qual:
        vec[word] += weights.mini_qual
    for word in doc.pref_qual:
        vec[word] += weights.pref_qual
    
    return dict(vec)  # convert back to a regular dict

def compute_query_tf(doc):
    vec = defaultdict(float)
    for word in doc:
        vec[word] += 1
    return dict(vec)

def compute_tfidf(doc, doc_freqs, weights, N, query=False):
    if query == True:
        vec_dict = compute_query_tf(doc)
    else:
        vec_dict = compute_tf(doc, weights)
    
    tfidf_vec = {}
    
    for word in vec_dict.keys():
        if word not in doc_freqs.keys():
            continue
        else:
            tfidf_vec[word] = vec_dict[word] * math.log(N / doc_freqs[word])
    return tfidf_vec


def compute_boolean(doc, doc_freqs, weights, N=0):
    # TODO
    vec = defaultdict(float)
    
    for word in doc.author:
        vec[word] = max(vec[word], weights.author)
    for word in doc.title:
        vec[word] = max(vec[word], weights.title)
    for word in doc.keyword:
        vec[word] = max(vec[word], weights.keyword)
    for word in doc.abstract:
        vec[word] = max(vec[word], weights.abstract)
    
    return vec


### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
        Computes the dot product of vectors x and y, represented as sparse dictionaries.
        '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)


def cosine_sim(x, y):
    '''
        Computes the cosine similarity between two sparse term vectors represented as dictionaries.
        '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


def dice_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return 2 * num / (sum(x.values()) + sum(y.values()))


def jaccard_sim(x, y):
    # return 0  # TODO: implement
    # if len(x) == 0 and len(y) == 0:
    #     return 1
    num = dictdot(x, y)
    denom = (sum(x.values()) + sum(y.values()) - num)
    if denom == 0:
        return 1
    return num / denom


def overlap_sim(x, y):
    # return 0  # TODO: implement
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / min(sum(x.values()), sum(y.values()))

### Extensions

# TODO: put any extensions here
def read_from_keyboard():
    flag = 'Yes'
    queries_text = []
    
    while flag == 'Yes':
        flag = input('Do you want to continue inputting query: (Yes/No)')
        if flag == 'No':
            break
        query = input('Please put in your self-introduction: ')
        queries_text += [query]
    
    return generate_queries(queries_text)


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
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = [[word for word in query if word not in stop_words] for query in processed_queries]

    # TODO: Add stem parameter
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
        output.append(docs[doc_id - 1])
    return output
