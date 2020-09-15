#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from itertools import product
import numpy

def find_text_similarity(str1, str2):

    ##---------------Defining stopwords for English Language---------------##
    stop_words = set(stopwords.words("english"))

    ##---------------Initialising Lists---------------##
    filtered_sentence1 = []
    filtered_sentence2 = []
    lemm_sentence1 = []
    lemm_sentence2 = []
    sims = []
    temp1 = []
    temp2 = []
    simi = []
    final = []
    same_sent1 = []
    same_sent2 = []
    #ps = PorterStemmer()

    ##---------------Defining WordNet Lematizer for English Language---------------##
    lemmatizer  =  WordNetLemmatizer()

    ##---------------Tokenizing and removing the Stopwords---------------##

    for words1 in word_tokenize(str1):
        if words1 not in stop_words:
            if words1.isalnum():
                filtered_sentence1.append(words1)

    ##---------------Lemmatizing: Root Words---------------##

    for i in filtered_sentence1:
        lemm_sentence1.append(lemmatizer.lemmatize(i))
    #print(lemm_sentence1)

    ##---------------Tokenizing and removing the Stopwords---------------##

    for words2 in word_tokenize(str2):
        if words2 not in stop_words:
            if words2.isalnum():
                filtered_sentence2.append(words2)

    ##---------------Lemmatizing: Root Words---------------##

    for i in filtered_sentence2:
        lemm_sentence2.append(lemmatizer.lemmatize(i))
    #print(lemm_sentence2)

    ##---------------Similarity index calculation for each word---------------##
    for word1 in lemm_sentence1:
        simi =[]
        for word2 in lemm_sentence2:
            sims = []
            #print(word1)
            #print(word2)
            syns1 = wordnet.synsets(word1)
            #print(syns1)
            #print(wordFromList1[0])
            syns2 = wordnet.synsets(word2)
            #print(wordFromList2[0])
            for sense1, sense2 in product(syns1, syns2):
                d = wordnet.wup_similarity(sense1, sense2)
                if d != None:
                    sims.append(d)
            #print(sims)
            #print(max(sims))
            if sims != []:        
               max_sim = max(sims)
               #print(max_sim)
               simi.append(max_sim)
        print(simi)
        if simi != []:
            max_final = max(simi)
            final.append(max_final)
    #print(final)

    ##---------------Final Output---------------##
    similarity_index = numpy.mean(final)
    similarity_index = round(similarity_index , 2)
    print("Similarity index value : ", similarity_index)

    if similarity_index >= 0.5:
        print("Similar")
        return True
    else:
        print("Not Similar")
        return False


# In[ ]:


# install new flask using command : pip install flask
import flask
from flask import request,jsonify
import json
app = flask.Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return '''<h1>Text Similarity</h1>
<p>Text Similarity API is used to find the similarity between two text.</p>'''
@app.route('/api/v1/text/match', methods=['GET'])
def textSimilarity():
    sentence1 = "";
    sentence2 = "";
    print(request.args)
    if 'sentence1' in request.args:
        sentence1 = request.args["sentence1"]
    else:
        return "Error: First sentence is not present in the request"
    if 'sentence2' in request.args:
        sentence2 = request.args["sentence2"]
    else:
        return "Error: Second sentence is not present in the request"
    isMatch = find_text_similarity(sentence1, sentence2)
    data = {}
    data["isMatch"] = isMatch
    print(data)
    return jsonify(data)
app.run()


# In[ ]:




