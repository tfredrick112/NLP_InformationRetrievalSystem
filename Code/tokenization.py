import re
import string
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# Split each sentence into words at spaces
		split_sentences = [sentence.split() for sentence in text]

		# Remove empty strings and also leading and trailing spaces and convert the word to lower case.
		# Also remove tokens that are stand-alone punctuation marks
		tokenizedText = [[word.strip().lower() for word in temp_list if (len(word.strip()) > 0 and not word in string.punctuation)] for temp_list in split_sentences]

		return tokenizedText


	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# Apply the Penn Treebank Tokenizer on all sentences
		tokenizedText = [TreebankWordTokenizer().tokenize(sentence) for sentence in text]

		# Convert all tokens to lower case and remove tokens that are stand-alone punctuation marks
		tokenizedText = [[word.lower() for word in sentence if not word in string.punctuation] for sentence in tokenizedText]

		return tokenizedText
