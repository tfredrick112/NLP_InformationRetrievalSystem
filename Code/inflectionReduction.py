import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		# Map from NLTK tags to Wordnet tags
		tag_map_dict = defaultdict(lambda : wordnet.NOUN, {"N": wordnet.NOUN, "J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV})

		# Initialize the WordNetLemmatizer
		wn_lemmatizer = WordNetLemmatizer()

		# Add the lemma of each word from each sentence into reducedText
		reducedText = [None]*len(text)
		for i, sentence in enumerate(text):
			# reducedText.append([wn_lemmatizer.lemmatize(word, return_WordNet_POS_tag(word)) for word in sentence])
			pos_tags = nltk.pos_tag(sentence)
			reducedText[i] = [wn_lemmatizer.lemmatize(word, tag_map_dict[tag[0]]) for word, tag in pos_tags]

		return reducedText
