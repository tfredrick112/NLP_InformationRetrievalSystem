import re
import nltk.data

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		sentences = re.split("(?<=[.!?])+", text) # Split at the .!? punctuation marks
		segmentedText = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0] # Remove leading and trailing spaces and empty strings
		return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		# Loading pre-trained PunktSentenceTokenizer
		punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		# Tokenize the text using Punkt Tokenizer
		segmentedText = punkt_tokenizer.tokenize(text)
		return segmentedText
