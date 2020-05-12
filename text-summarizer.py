"""
This is source code to test the algorithm of Synogen, and determine its score
by taking the similarity between machine generated synopsis and manually generated one
"""

#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
from os import listdir
from os.path import isfile, join

nltk.download("stopwords")
nltk.download('punkt')
stop_words = stopwords.words('english')

def crawlFolder(path):
    """
    crawling the dataset
    return list which contain the information of dataset, [content of dataset, path to dataset]
    """
    path ="dataset/plot/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    dataset = []
    for item in files:
        file_name = join(path, item)
        file = open(file_name, "r")
        data = file.read()
        file.close()
        dataset.append([data, file_name])
    return dataset

def read_article(file):
    """
    read an article (file), split it to sentences and tokenize it
    return tokenized file
    """
    article = file[0].split(". ")
    sentences = []
    for sentence in article:
        if len(sentence) <2 :
            continue
        sentences.append(nltk.tokenize.word_tokenize(sentence))
        
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    """
    calculate the similarity between sentence!
    return distance between sentences
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
    make a matrix to plot the similarity between sentences in a file
    return matrix
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
    return summarized text
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

"""
getting the dataset
"""

plot = crawlFolder("dataset/plot")
synopsis = crawlFolder ("dataset/synopsis")
# let's begin

"""
creating the summary
"""
a = 0
summary = []
for item in plot:
    print("generating summary for document "+ plot[a][1])
    a+=1
    suma = generate_summary(item)
    summary.append(suma)

"""
determine scores
"""

score = []

#do this python -m spacy download en
import spacy
nlp = spacy.load('en')
for i in range(len(plot)):
    doc1 = nlp(summary[i])
    doc2 = nlp(synopsis[i][0])
    score.append(doc1.similarity(doc2))

"""
make a report of scores
"""

print("=========================================================")

for i in range(len(score)):
    print ("score for document "+ plot[i][1] + " is : " + str(score[i]))

print("=========================================================")

print("overall distance to generated summary with manual created summary :" + str(sum(score)/len(score)))

def printDocument(document, filename):
    filename = filename[1]
    filename = filename.split("/")
    filename = filename[-1]
    filename = join("output", filename)
    fp = open(filename, "w")
    fp.write(document)
    fp.close()

for i in range (len (summary)):
    printDocument(summary[i], plot[i])