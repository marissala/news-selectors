"""
Created on Wed Feb  6 11:16:12 2019
@author: marissala
"""

import nltk
import spacy
import pandas as pd

from success_algorithm import success
from NERTokenizer import NERTokenizer
from CustomTFIDF import CustomTFIDF


#########################################################
#################### HAC ################################
#########################################################

df = pd.read_csv("~/Documents/Sentinel/news-selectors/results_event_perline_repeated_linerepeated.csv", sep=",", encoding="utf-8")
df['date'] = 0
new_df = df
corpus = new_df['content'].tolist()

# Tokenizing
NerTok = NERTokenizer(tag=True)
toks = NerTok.transform(corpus)

# Vectorizing, get the TFIDF
Vectorizer = CustomTFIDF(ents_rate = 6.368, person_rate = 2.263, julian = False)
matrix= Vectorizer.transform(toks)

# Train topic number estimator
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
hdp = HdpModel(common_corpus, common_dictionary)

# Infer topic distribution
unseen_document = [(1, 3.), (2, 4)]
doc_hdp = hdp[unseen_document]

# Print 20 topics with 10 most probable words
topic_info = hdp.print_topics(num_topics=5, num_words=10)

# Update model with new topics coming in
hdp.update([[(1, 2)], [(1, 1), (4, 5)]])

# not sure how the updating works so far
#hdp.update(["hello there"], ["another"])

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters=3, affinity = "euclidean")
hac.fit(matrix)
#dense_matrix = hac.fit(matrix).todense()

# Predicted success based on the labels made
y_pred = hac.labels_.tolist()
# In the success function there is a place where we read in data and manipulate the "true" labels we compare against
success(df, hac, y_pred, matrix)

#clusters = y_pred