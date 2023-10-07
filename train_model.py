import multiprocessing
from gensim.models import Word2Vec

def train_w2v(w2v_df):
    cpu_int = multiprocessing.cpu_count()
    #the following parameters can be changed if needed
    w2v_model= Word2Vec(min_count=4, window=4,vector_size=100,alpha=0.03,min_alpha=0.0007,sg=1,workers=cpu_int-1)
    #For build_vocab, we have to create a list where each element is a corpus. A corpus is a list of words or phrases
    content = [sentence.split(" ") for sentence in w2v_df.values]
    w2v_model.build_vocab(content,progress_per=100)
    w2v_model.train(content, total_examples=w2v_model.corpus_count, epochs=1, report_delay=1)
    return w2v_model
