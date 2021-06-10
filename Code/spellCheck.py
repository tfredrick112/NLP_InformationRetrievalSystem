import json
import string
import re
from time import time
from collections import Counter

total_words = []
vocabulary = []
word_counts = None
docs = []


def Pword(word):
    """
    Probability of the word in the corpus
    """
    global word_counts

    return word_counts[word]/total_words

def Pword_bigram(word, previous_word):
    """
    Probability of the word in the corpus
    """
    global docs, word_counts

    bigram_count = 0
    for doc in docs:
        bigram_count += doc.count(previous_word + " " + word)

    p2 = bigram_count/word_counts[previous_word]

    return p2


def one_edit(word):
    """
    This functions returns all words at edit distance 1. For w word of length n:
    Insertions: 26(n + 1)
    Deletions: n
    Transpositions: n- 1
    Substitutions: 26n
    Total candidates: 54n + 25
    """
    letters = string.printable[10:36]
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def two_edits(word):
    """
    Returns all the words that are a distance of two edits from a word.
    """
    list_of_edits = []
    for e in one_edit(word):
        # We find all words at edit distance 1, from those words which are edit distance from the original word
        # to get words at edit distance 2
        list_of_edits = list_of_edits + list(one_edit(e))

    return set(list_of_edits)

def correct_word(word):
    """
    Returns the corrected form of the word.
    """
    global vocabulary

    # Generates the candidates at edit distance 1 and 2
    start = time()
    # Of all the words that are one edit away, those that exist in the corpus
    words_one_edit_away = one_edit(word)
    candidates1 = [w for w in words_one_edit_away if w in vocabulary]

    # Of all the words that are two edits away, those that exist in the corpus
    words_two_edits_away = two_edits(word)
    candidates2 = [w for w in words_two_edits_away if w in vocabulary]

    stop = time()
    print("Time to generate candidates = {}".format(stop-start))

    scores1 = [(Pword(w), w) for w in candidates1]
    if scores1:
        max_score, best_word = max(scores1)
        return best_word

    scores2 = [(Pword(w), w) for w in candidates2]
    if scores2:
        max_score, best_word = max(scores2)
        return best_word

    # In the worst case, return the word itself
    return word

def doSpellCheck(query, documents):
    """
    Returns the corrected query.
    """
    global all_words, total_words, vocabulary, docs, word_counts
    all_words = []
    docs = documents

    # List of all words in all documents
    for doc in docs:
        all_words = all_words + re.sub("[^a-z ]+", " ", doc).split()

    # Total number of words in the corpus
    total_words = len(all_words)

    # A list of all the unique words in the corpus
    vocabulary = list(set(all_words))

    # Counts of each word
    word_counts = Counter(all_words)

    # Corrected query
    new_query = ""

    for word in query.split():
        if word not in vocabulary:
            new_query += " " + correct_word(word)
        # If the word itself exists in the dictionary, return the same word
        else:
            new_query += " " + word

    return new_query.strip()
