#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:50:28 2019
@author: parkerglenn
https://github.com/parkervg/news-article-clustering/blob/master/clustering.py
"""
import os
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from NERTokenizer import NERTokenizer
from CustomTFIDF import CustomTFIDF
from success_algorithm import success

"""
Creating relevant classes
"""
NerTok = NERTokenizer(tag=True)
Vectorizer = CustomTFIDF(ents_rate = 6.368, person_rate = 2.263, julian = False)
stemmer = SnowballStemmer("english")

"""
Cleaning DF
"""
#os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")
#df = pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv")
#labels_df= pd.read_csv("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_ClassificationFINAL.csv")
#Deletes unnecessary columns
#df = df.drop(df.columns[:12], axis = 1)
#Sets manageable range for working data set
new_df = df#[5000:6000]
#Gets info in list form to be later called in kmeans part
corpus = new_df['content'].tolist()
titles = new_df["title"].tolist()
#labels_df starts at df[5000] so we're good on the matching of labels to content
labels_df = df
events = labels_df["event"].tolist()#[:1000]
links = new_df["title"].tolist() #M: should be url

clusters = y_pred

"""
Creating matrix
"""
toks = NerTok.transform(corpus)
matrix= Vectorizer.transform(toks)

"""
Clustering and measuring success.
"""
#########################################################
####################BIRCH################################
#########################################################
from sklearn.cluster import Birch
brc = Birch(n_clusters = 20)
brc.fit(matrix)

y_pred = brc.labels_.tolist()
success(df, df, brc, y_pred, matrix)


#########################################################
####################HAC##################################
#########################################################
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters=52, affinity = "euclidean")
hac.fit(matrix)
#dense_matrix = tfidf_matrix.todense()

#from sklearn.externals import joblib
#Saves the model you just made
#joblib.dump(hac, '350_euc_HAC_ENTS.pkl')
#hac = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/HAC_Cluster_Models/350_euc_HAC.pkl")

y_pred = hac.labels_.tolist()
success(df, df, hac, y_pred, matrix)

#########################################
# Inserting some code here from http://brandonrose.org/clustering
# He defines terms somewhere here and I need them later I think
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

#define vectorizer parameters
import re
tfidf_vectorizer = TfidfVectorizer(max_df = 2000, max_features=200000, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3)) #max_df=0.8, min_df=0.1
%time tfidf_matrix = tfidf_vectorizer.fit_transform(corpus) #fit the vectorizer to synopses

terms = tfidf_vectorizer.get_feature_names()
#terms = Vectorizer.get_feature_names()
#########################################


#########################################################
####################KEMANS###############################
#########################################################
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters = num_clusters)
km.fit(matrix)

y_pred = km.labels_.tolist()

#joblib.dump(km,  'doc_cluster.pkl')

#km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

success(df, df, km, y_pred, matrix)





#########################################################
###############KMEANS CLUSTER EXPLORING##################
#########################################################
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

import re
def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in corpus:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

#Let's you search with stemmed word to see original format of word
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
# MARIS: updating here because the for loop below needs "terms"
#vocab_frame =
#vocab_frame.reset_index(drop=False).rename(columns={"words":"terms"})
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


articles = {"title": titles, "date": new_df["date"], "cluster": clusters, "content": new_df["content"], "event": events[:1000]}
frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'date', 'cluster', 'content', "event"])
frame['cluster'].value_counts()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]



########################################################
########################################################
########################################################
# This is maybe not so useful to debug - instead debug the success_algorithm y_predDict emptiness!

from collections import Counter
#Creates a count dict (success) to see how many instances of the same event are clustered together
for i in clusters[:100]:
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print()
    counts = []
    for event in frame.loc[i]["event"].values.tolist():
        counts.append(event)
    counts = dict(Counter(counts))
    print(counts)
    print()
    print()


#Allows you to zoom in on a specific cluster, see what words make that cluster unique
for i in clusters:
    if i == 1: #Change 2 to the cluster
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :20]: #replace 20 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        counts = []
        for event in frame.loc[i]["event"].values.tolist():
            counts.append(event)
        counts = dict(Counter(counts))
        print(counts)
        print()
        print()




# FROM http://brandonrose.org/clustering
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')

    # Disentangling the events per article
    #for j in range(len(frame.loc[i]["event"].values)):
    #    for event in frame.loc[i]["event"].values:
    #        print(event.split(', ')[j])
    
    df = pd.DataFrame(frame.loc[0]["event"].values).reset_index(drop=True).rename(columns={0: "words"})
    ff = df.apply(disentangle, axis = 1)
    
    counts = []
    for event in frame.loc[i]["event"].values.tolist():
        counts.append(event)
    #for event in ff.values.tolist():#.tolist():
    #    counts.append(event)
    

    #for event in frame.loc[i]["event"].values.tolist():
    #    print(event)
    #    if len(event) > 1:
    #        #for j in range(len(event)):
    #        ev = event.split(', ')
    #        counts.append(ev)
    #counts = dict(Counter(counts))
    print() #add whitespace
    print() #add whitespace



####################################################
###################################################
#####################################################

# Disentangling the events per article
for j in range(len(frame.loc[i]["event"].values)):
    for event in frame.loc[i]["event"].values:
        print(event.split(', ')[j])

# trying out with a row-wise function but it works badly af?
def disentangle(row):
    return(str(row["words"]).split(', '))

df = pd.DataFrame(frame.loc[0]["event"].values).reset_index(drop=True).rename(columns={0: "words"})
ff = df.apply(disentangle, axis = 1)





###################################################
## MULTIDIMENSIONAL SCALING

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'
                 }

def random_dict(n):
    import random
    from random import randrange
    values = ['value1', 'value2']
    mydict = {i: values[0] for i in range(n)}
    mydict[random.randrange(n)] = values[1]
    return mydict

random_dict(10)

cluster_names = random_dict(58)

#some ipython magic to show the matplotlib plots inline
%matplotlib inline 

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  

    
    
plt.show() #show the plot