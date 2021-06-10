import nltk
from nltk.corpus import stopwords

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		# Use a predefined list of stop words
		stopwordslist = stopwords.words('english')

		# From each sentence (list of words), filter out stop words
		stopwordRemovedText = [[word for word in sentence if word not in stopwordslist] for sentence in text]
		return stopwordRemovedText
