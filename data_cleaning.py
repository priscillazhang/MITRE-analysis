import glob
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import PerceptronTagger
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

#Download Resources
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("wordnet")


class DataCleaning():
    
    def _dataPreprocessing(self, doc):
        
        '''
        Preprocessing the data
        '''
        
        # Put all the texts in lower case
        modComm = doc.lower()
        # Remove @ tags 
        at_tags=' '.join(re.findall(r'(?<=@)[^\s]+\s?',modComm))
        modComm=re.sub(r'@.*?\s','',modComm)
        

        return modComm

    def CorpusListWords(self, matrix):

        '''
        Gives the noun pharases after preprocessing the data
        '''

        tokenizer = RegexpTokenizer(r'\w+')
        word_list = []        

        for doc in matrix:
            cleaned_data = self._dataPreprocessing(doc)
            blob = TextBlob(str(cleaned_data))
            noun = blob.noun_phrases
            word_list.append(",".join(noun))

        return word_list
    
    def _leaves(self,tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP' or t.label()=='JJ' or t.label()=='RB'):
            yield subtree.leaves()

    def _getTerms(self,tree):
        for leaf in self._leaves(tree):
            term = [w for w,t in leaf ]
            # Phrase only
            if len(term)>1:
                yield term

    # Flatten phrase lists to get tokens for analysis
    def _flatten(self, npTokenList):
        finalList =[]
        for phrase in npTokenList:
            token = ''
            for word in phrase:
                token += word + ' '
            finalList.append(token.lower().rstrip())
        return finalList
    
    def CorpusListPhrase(self, matrix, stopwords):

        phrase_list=[]
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
                
            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """

        tagger = PerceptronTagger()
        pos_tag = tagger.tag
        # Create phrase tree
        chunker = nltk.RegexpParser(grammar)
        for doc in matrix:
                phrase=self._flatten([word for word in 
                                self._getTerms(chunker.parse(pos_tag(re.findall(r'\w+',str(doc))))) 
                                if word not in stopwords])
                phrase_list.append(",".join(phrase))

        return phrase_list
