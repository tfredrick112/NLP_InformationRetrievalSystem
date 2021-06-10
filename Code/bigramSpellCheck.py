import itertools
from Levenshtein import distance
import os
import string

import re

class BigramSpellCheck:
    def __init__(self, docs):
        all_words = []
        for doc in docs:
            all_words = all_words + re.sub("[^a-z ]+", " ", doc).split()

        # Total number of words in the corpus
        total_words = len(all_words)

        # A list of all the unique words in the corpus
        self.vocabulary = list(set(all_words))

        # if not os.path.exists("Output/vocabulary.txt"):
        #     with open("Output/vocabulary.txt", "w+") as f:
        #         for w in self.vocabulary:
        #             f.write(w+"\n")

        # Generate all possible bigrams from aa to zz
        self.all_bigrams = ["".join(tup) for tup in itertools.product(string.ascii_lowercase, repeat=2)]

        # Create the bigram inverted index
        self.bigram_inverted_index = {}
        for bigram in self.all_bigrams:
            self.bigram_inverted_index[bigram] = [word for word in self.vocabulary if bigram in word]


    def correct_word(self, query_word):
        """
        Returns the corrected word.
        """

        if query_word in  self.vocabulary:
            return query_word

        # Generate all the bigrams in the query word
        query_bigrams = [query_word[i:i+2] for i in range(len(query_word)-1)]

        # Get the candidate corrections
        candidates = []
        for qb in query_bigrams:
            candidates = candidates + self.bigram_inverted_index[qb]

        # Remove duplicates
        candidates = list(set(candidates))

        # A better candidate is one that has at least 40% of the bigrams in the query word
        better_candidates = []
        for candidate in candidates:
            candidate_bigrams = [candidate[i:i+2] for i in range(len(candidate)-1)]
            common_bigrams = [bigram for bigram in query_bigrams if bigram in candidate_bigrams]

            if len(common_bigrams)/len(query_bigrams) >= 0.5:
                better_candidates.append(candidate)

        if len(better_candidates)==0:
            for candidate in candidates:
                candidate_bigrams = [candidate[i:i+2] for i in range(len(candidate)-1)]
                common_bigrams = [bigram for bigram in query_bigrams if bigram in candidate_bigrams]

                if len(common_bigrams)/len(query_bigrams) >= 0.1:
                    better_candidates.append(candidate)

        if len(better_candidates)==0:
            return query_word

        # Compute edit distances to all the candidates and pick the nearest word
        edit_distances = []

        for candidate in better_candidates:
            edit_distances.append((distance(query_word, candidate), candidate))

        edit_distances.sort()
        return edit_distances[0][1]

    def correct_query(self, query):
        """
        Returns the corrected query.
        """
        # Corrected query
        new_query = ""

        for word in query.split():
            if word not in self.vocabulary:
                new_query += " " + self.correct_word(word)
            else:
                new_query += " " + word

        return new_query.strip()
