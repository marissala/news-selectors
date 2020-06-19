# from PreProcessing.NerTokenizer import NERTokenizer
# Source: https://github.com/parkervg/news-article-clustering/blob/master/PreProcessing/NERTokenizer.py

n = "B"  # you can set this too, but this is optional
def find_letter(lst):
    # do not set a value to lst in here

    if not lst:          
        return 0

    elif lst[0] == n:  # You checked first element in here
        return True

    elif find_letter(lst[1:]):  # since you checked the first element, skip it and return the orher elements of the list
        return True

    else: 
        return False

import sklearn.base
class NERTokenizer(sklearn.base.TransformerMixin):
    """If 'tag' is True, Person entities .startswith("*") and other entities deemed "good" .startswith("&")"""
    def __init__(self, tag = False):
        self._tag = tag

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer("english")

        import spacy
        from spacy.gold import iob_to_biluo
        #nlp = #spacy.load('C:/Users/maris/Anaconda3/envs/SBERT-WK/lib/site-packages/spacy/data/en/en_core_web_sm/en_core_web_sm-2.0.0', disable=['parser','tagger','textcat'])
        # This doesn't work for me (linking up error)
        #nlp = spacy.load('en_core_web_md', disable=['parser','tagger','textcat'])
        #import en_core_web_md
        #nlp = en_core_web_md.load(disable=['parser','tagger','textcat'])
        nlp = spacy.load('en')#, disable=['parser','tagger','textcat'])
        from spacy.attrs import ORTH
        nlp.tokenizer.add_special_case("I'm", [{ORTH: "I'm"}])
        nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

        english_stopwords = stopwords.words('english')
        english_stopwords.append("i'm")

        tokenized_corpus = []
        good_ents = ["PERSON","GPE","ORG", "LOC", "EVENT", "FAC"]
        continue_tags = ["B-","I-"]
        end_tags = ["L-","U-"]



        for text in X:
            toks = []
            iobs = [i.ent_iob_ for i in nlp(text)]
            print(iobs)
            """
            lst = ['B', iobs, 1]
            if find_letter(lst):
                new_iobs = []
                for string in iobs:
                    new_string = string.replace("B", "O")
                    new_iobs.append(new_string)
                iobs = new_iobs
                iobs[0] = "B"
                print(iobs)
                biluos = list(iob_to_biluo(iobs))
            else:
                biluos = list(iob_to_biluo(iobs))
            """

            ### MY CODE
            #new_iobs = []
            #for string in iobs:
            #    new_string = string.replace("B", "O")
            #    #new_string = string.replace("I", "O")
            #        #print("new: ", new_string)
            #    new_iobs.append(new_string)
            #new_iobs2 = []
            #for string2 in new_iobs:
            #    new_string2 = string2.replace("I", "O")
            #    new_iobs2.append(new_string2)
            ####


            #biluos = list(iob_to_biluo(iobs))
            #Named entities variable
            ne = ""
            for index, tok in enumerate(nlp(text)):
                if iobs[index] in continue_tags and str(tok.ent_type_) in good_ents:
                    #str(tok).split() != [] Checks if empty token
                    #For some reason tok.whitespace_ doesn't include double token entities
                    #like "JENNIFER LAWRENCE"
                    if not self._tag:
                        ne += " " + str(tok).lower()
                    elif self._tag and str(tok).split() != []:
                        #Entity is the beginning of an entity set
                        if iobs[index] == "B-":
                            if str(tok.ent_type_) != "PERSON":
                                ne += " &" + str(tok).lower()
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " *" + str(tok).lower()
                        else:
                            if str(tok.ent_type_) != "PERSON":
                                ne += " " + str(tok).lower()
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " " + str(tok).lower()
                elif iobs[index] in end_tags and str(tok.ent_type_) in good_ents:
                    if not self._tag:
                        ne += " " + str(tok).lower()
                        toks.append(ne.lstrip())
                        ne = " "
                    elif self._tag and str(tok).split() != []:
                        #Entity is just a single unit
                        if iobs[index] == "U-":
                            if str(tok.ent_type_) != "PERSON":
                                ne += " &" + str(tok).lower()
                                toks.append(ne.lstrip())
                                ne = " "
                            elif str(tok.ent_type_) == "PERSON":
                                ne += " *" + str(tok).lower()
                                ne.replace("*’m", "")
                                toks.append(ne.lstrip())
                                ne = " "
                        else:
                            ne += " " + str(tok).lower()
                            # so that possesive tags are not stored with the '’s'
                            ne = ne.replace("’s", "")
                            toks.append(ne.lstrip())
                            ne = " "
                #If token is just a boring old word
                else:
                    if not tok.is_punct and not tok.is_space and str(tok).lower() not in english_stopwords:
                        toks.append(stemmer.stem(str(tok)))
            tokenized_corpus.append(toks)
        return tokenized_corpus