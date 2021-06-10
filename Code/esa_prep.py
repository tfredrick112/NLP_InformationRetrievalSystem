import wikipedia as wp
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict, Counter
import pandas as pd
import json
import os
import numpy as np

if not os.path.exists("Output/article_dict.json"):
    # This is the file that we got from Petscan
    df = pd.read_csv("Large_files/download")
    # Get all the page ids
    pageid_list = list(df["pageid"])

    article_dict = {}

    # Map from NLTK tags to Wordnet tags
    tag_map_dict = defaultdict(lambda : wordnet.NOUN, {"N": wordnet.NOUN, "J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV})
    # Initialize the WordNetLemmatizer
    wn_lemmatizer = WordNetLemmatizer()

    for pid in pageid_list:
        try:
            # Get content of the page using the page id
            cont = wp.page(pageid=pid).content

            cont = cont[:cont.find("See also")].lower()

            # Remove the Latex content
            filt1 = re.sub("\{.*", " ", cont)
            filt2 = re.sub("[^a-z ]+", " ", filt1)

            # Get stopwords in English
            stop_list = stopwords.words("english")

            # Split into words, remove stopwords and words with less than 3 characters.
            temp = [word for word in filt2.split() if (word not in stop_list and len(word) > 2)]

            pos_tags = nltk.pos_tag(temp)
            article_dict[pid] = Counter([wn_lemmatizer.lemmatize(word, tag_map_dict[tag[0]]) for word, tag in pos_tags])

        except:
            pass


    # Save dictionary in JSON file
    with open('Output/article_dict.json', 'w') as fp:
        json.dump(article_dict, fp)

if not os.path.exists('Output/esa_vectors.json'):
    with open('Output/article_dict.json', 'r') as fp:
        article_dict = json.load(fp)
        

    with open("Output/vocabulary.txt", "r") as f:
        vocab = f.readlines()
    vocab = [word[:-1] for word in vocab]

    esa_vectors = {}
    pageid_list = list(article_dict.keys())

    esa_doc_lengths = {}
    for pid in pageid_list:
        esa_doc_lengths[pid] = sum(article_dict[pid].values())

    for word in vocab:
        #esa_vectors[word] = np.zeros(len(pageid_list), dtype="float")
        esa_vectors[word] = [0]*len(pageid_list)
        for i, pid in enumerate(pageid_list):
            if word in article_dict[pid]:
                esa_vectors[word][i] = article_dict[pid][word]/esa_doc_lengths[pid]

    with open('Output/esa_vectors.json', 'w') as fp:
        json.dump(esa_vectors, fp)
