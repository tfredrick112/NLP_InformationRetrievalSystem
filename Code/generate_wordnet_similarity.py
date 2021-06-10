# Code for generating Wu-Palmer WordNet similarity scores
# The similarity scores have been pre-computed in the Large_files/wordnet_sim_arr.npy file
# Hence, this code need not be run. Only for verification purpose

from nltk.corpus import wordnet
import os, json
import numpy as np

# Get all vocabulary words
if os.path.exists('Large_files/inverted_index.json'):
    with open('Large_files/inverted_index.json', 'r') as fp:
        index = json.load(fp)
        
terms = list(sorted(index.keys()))

wordnet_sim_arr = np.zeros((len(terms), len(terms)))
for i in range(len(terms)):
    if i%100==0:
            print(i)
    syn_i = wordnet.synsets(terms[i]) # Obtain synsets of each term
    if len(syn_i)==0:
        wordnet_sim_arr[i][i]=1
        continue
    for j in range(len(terms)):
        syn_j = wordnet.synsets(terms[j])
        if len(syn_j)==0:
            continue
        wordnet_sim_arr[i][j] = syn_i[0].wup_similarity(syn_j[0]) # Find similarity

# Store similarity scores in an array
np.save("Large_files/wordnet_sim_arr.npy", wordnet_sim_arr)