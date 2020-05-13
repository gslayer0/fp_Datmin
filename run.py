#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:30:55 2019

@author: temperantia
"""
from flask import Flask
from flask import request
from flask import render_template
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
app = Flask(__name__)

nltk.download("stopwords")
stop_words = stopwords.words('english')

def read_article(file):
    """
    read an article (the plot that provided by user)
    """
    article = file.split(". ")
    sentences = []
    for sentence in article:
        if len(sentence) <2 :
            continue
        sentences.append(nltk.tokenize.word_tokenize(sentence))
        
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    """
    calculate similarity between two sentence
    """
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    """
    build a matrix that map the similarity of the sentence in a file
    """
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(data, top_n=5):
    """
    generating summary
    """
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(data)
    for item in sentences:
        if len(item)<3:
            sentences.remove(item)

    # indexing unranked
    dict_of_unranked_sentences = {}
    keys = range(len(sentences))
    # values = ["Hi", "I", "am", "John"]
    for i in keys:
        dict_of_unranked_sentences[i] = sentences[i]
    # print(dict_of_unranked_sentences)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    
    ranked_sentence = []
    for i in range (len(scores)):
        ranked_sentence.append([scores[i], sentences[i], i])
    
    ranked_sentence.sort(key = lambda x: x[0], reverse=True)
    summary = ranked_sentence[0:top_n]
    summary.sort(key = lambda x: x[2])
    summary = [i[1] for i in summary]
    summarize_text = ""
    for i in range(top_n):
        summarize_text+=" ".join(summary[i]) + ". "

    # Step 5 - Offcourse, output the summarize texr
    return summarize_text


@app.route("/", methods=['GET', 'POST'])
def aaa():
    """
    serving the web
    """
    if request.method == 'GET':
        return render_template("index.html",flag=0, summary = "")
    else:
        text = request.form['text']
        N = request.form['N']
        N = int(N)
        print (N)
        summary = generate_summary(text, N)
         
        return render_template("index.html", flag=1, summary = summary, text=text, N=N)


if __name__ == '__main__':
   app.run()
