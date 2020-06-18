import nltk
import spacy
import pandas as pd

#from success.success_algorithm import success
from NERTokenizer import NERTokenizer
from CustomTFIDF import CustomTFIDF

#########################################################
#################### HAC ################################
#########################################################

df = pd.read_csv("~/Documents/Sentinel/news-selectors/fake_data.csv", sep=";")
new_df = df
corpus = new_df['content'].tolist()

# Tokenizing
NerTok = NERTokenizer(tag=True)
toks = NerTok.transform(corpus)

# Vectorizing, get the TFIDF
Vectorizer = CustomTFIDF(ents_rate = 6.368, person_rate = 2.263, julian = False)
matrix= Vectorizer.transform(toks)

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters=2, affinity = "euclidean")
hac.fit(matrix)
#dense_matrix = tfidf_matrix.todense()

#from sklearn.externals import joblib
#Saves the model you just made
#joblib.dump(hac, '350_euc_HAC_ENTS.pkl')
#hac = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/HAC_Cluster_Models/350_euc_HAC.pkl")

# Predicted success based on the labels made
y_pred = hac.labels_.tolist()
# In the success function there is a place where we read in data and manipulate the "true" labels we compare against
success(hac, y_pred, matrix)