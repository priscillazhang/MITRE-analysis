import gensim
import numpy as np
from gensim import corpora, models
from gensim.matutils import corpus2csc
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


class FeatureSelection:

    def tfidfTransformation(self, matrix, min_num_of_words=1, max_num_of_words=3):
        """
        Perform tf-idf metrics on corpus for clustering
        """
        features_tfidf = TfidfVectorizer(ngram_range=(min_num_of_words, max_num_of_words),analyzer=u'word', sublinear_tf=True, use_idf=True,
                                      lowercase=True, stop_words='english').fit_transform(matrix)
        
        return features_tfidf

    def CSRMatrixConverter(self, features_TF):
        np.savez('tf-idf_matrix',data = features_TF.data,
                 indices=features_TF.indices,indptr =features_TF.indptr, shape=features_TF.shape)
        loader = np.load('tf-idf_matrix.npz')
        Fphrase_tfidf=csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                 shape = loader['shape'])
        return Fphrase_tfidf
    
    def BoWTransformation(self, matrix):
        
        split_matrix = [doc.split(',') for doc in matrix]
        
        dictionary = gensim.corpora.Dictionary()
        bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in split_matrix]
        
        return bow_corpus
    
    # def tf_idf_lda(self):
    #     bow_corpus=self.bow_lda()
    #     tfidf = models.TfidfModel(bow_corpus)
    #     corpus_tfidf = tfidf[bow_corpus]
    #     return corpus_tfidf
    
    def Word2Vec(self, matrix, hyper_tune=False, parameters={}, size=300, window=3, min_count=1, workers=4):
        """
        Perform word2vec metrics on corpus
        """
        model = Word2Vec(matrix, size=size, parameters=parameters, window=window, min_count=min_count, workers=workers)
        if hyper_tune:
            model = GridSearchCV(model, parameters)
        f_phrase_w2v=model.wv       
            
        return f_phrase_w2v.vectors
        
    def Doc2Vec(self, matrix, vector_size=80, window=2, min_count=1, workers=4):
        split_matrix = [doc.split(',') for doc in matrix]
        documents = [TaggedDocument(doc, [ind]) for ind, doc in enumerate(split_matrix)]
        model_doc2vec = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        fname = get_tmpfile("doc2vec_model")
        model_doc2vec.save(fname)
        model_doc2vec = Doc2Vec.load(fname)
        return model_doc2vec.docvecs.vectors_docs
    
    def Perform(self,matrix, func_name): # The function name is either "tfidf" or "bow"
        if func_name == 'tfidf':
            vsm_corpus = self.tfidfTransformation(matrix)
        elif func_name == 'bow':
            vsm_corpus = self.BoWTransformation(matrix)
        return vsm_corpus
