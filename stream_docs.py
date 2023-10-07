from gensim.corpora import Dictionary
import gensim
import os
dct = Dictionary(["máma mele maso".split(), "ema má máma".split(),["máma"]])
print(dct.doc2bow(["this", "is", "máma"]))
print("máma mele maso".split())
print(dct.doc2bow(["this", "is", "máma","máma"], return_missing=True))

def iter_documents(top_directory):
    """
    Generator: iterate over all relevant documents, yielding one
    document (=list of utf8 tokens) at a time.
    """
    print("in iter_doc")
    # find all .txt documents, no matter how deep under top_directory
    for root, dirs, files in os.walk(top_directory):
        for fname in filter(lambda fname: fname.endswith('.txt'), files):
            # read each document as one big string
            document = open(os.path.join(root, fname),'rb').read()
            # break document into utf8 tokens
            yield gensim.utils.tokenize(document, lower=True, errors='ignore')
 
class TxtSubdirsCorpus(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, top_dir):
        self.top_dir = top_dir
        print("initializing")
        # create dictionary = mapping for documents => sparse vectors
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir))
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        print("iterating")
        for tokens in iter_documents(self.top_dir):
            # transform tokens (strings) into a sparse vector, one at a time
            yield self.dictionary.doc2bow(tokens)
 
# that's it! the streamed corpus of sparse vectors is ready
corpus = TxtSubdirsCorpus(".\\test_data")
 
# print the corpus vectors
for vector in corpus:
    print(vector)
 
# # or run truncated Singular Value Decomposition (SVD) on the streamed corpus
from gensim.models.lsimodel import stochastic_svd as svd
u, s = svd(corpus, rank=200, num_terms=len(corpus.dictionary), chunksize=5)
print(f"{u} {s}")