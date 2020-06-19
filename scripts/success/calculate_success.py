import nltk
import spacy
import pandas as pd

from success_algorithm import success
from NERTokenizer import NERTokenizer
from CustomTFIDF import CustomTFIDF

#########################################################
#################### HAC ################################
#########################################################

#df = pd.read_csv("~/Documents/Sentinel/news-selectors/fake_data.csv", sep=";")
df = pd.read_csv("~/Documents/Sentinel/news-selectors/results_event_perline.csv", sep=",", encoding="utf-8")
df['date'] = 0
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
hac = AgglomerativeClustering(n_clusters=23, affinity = "euclidean")
hac.fit(matrix)
#dense_matrix = hac.fit(matrix).todense()

#from sklearn.externals import joblib
#Saves the model you just made
#joblib.dump(hac, '350_euc_HAC_ENTS.pkl')
#hac = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/HAC_Cluster_Models/350_euc_HAC.pkl")

# Predicted success based on the labels made
y_pred = hac.labels_.tolist()
# In the success function there is a place where we read in data and manipulate the "true" labels we compare against
success(df, df, hac, y_pred, matrix)


clusters = y_pred

# df has 'content' --> lets make TFIDF etc
# labels_df has 'event'

### NEED TO GET 'event' and 'content' and 'title' columns out for the Polish data!


import en_core_web_md
nlp = en_core_web_md.load(disable=['parser','tagger','textcat'])

from spacy.gold import iob_to_biluo

from spacy.attrs import ORTH
nlp.tokenizer.add_special_case("I'm", [{ORTH: "I'm"}])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

for text in corpus:
    toks = []
    iobs = [i.ent_iob_ for i in nlp(text)]
    print(iobs)
    #biluos = list(iob_to_biluo(iobs))



list(iob_to_biluo(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']))





##########################
for text in corpus:
    iobs = [i.ent_iob_ for i in nlp(text)]
    biluos = list(iob_to_biluo(iobs))



    new_iobs = []
    for string in iobs:
        new_string = string.replace("B", "O")
        #new_string = string.replace("I", "O")
            #print("new: ", new_string)
        new_iobs.append(new_string)
    new_iobs2 = []
    for string2 in new_iobs:
        new_string2 = string2.replace("I", "O")
        new_iobs2.append(new_string2)

    biluos = list(iob_to_biluo(new_iobs2))


nlp = en_core_web_md.load()
from spacy.attrs import ORTH


import spacy
from spacy.gold import iob_to_biluo
nlp = spacy.load('en', disable=['parser','tagger','textcat'])
#nlp.tokenizer.add_special_case("I'm", [{ORTH: "I'm"}])
#nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
doc = 'My name is George Washington Singer, and I am one an englishman' #US President Donald Trump has attacked Democrats impeachment investigation'
print([i.ent_iob_ for i in nlp(doc)])
iobs = [i.ent_iob_ for i in nlp(doc)]
iob_to_biluo(iobs)


doc = 'President Donald Trump has attached democrats'# impeachment investigation'
iobs = [i.ent_iob_ for i in nlp(doc)]
iob_to_biluo(iobs)

doc = "US President Donald Trump has attacked Democrats impeachment investigation" #into his conduct as very unpatriotic"
print([i.ent_iob_ for i in nlp(doc)])

doc = 'US President Donald Trump has attacked Democrats\' impeachment investigation into his conduct as "very unpatriotic".'

iobs = [i.ent_iob_ for i in nlp(doc)]
iob_to_biluo(iobs)

for i in iobs:
    print(len(i))