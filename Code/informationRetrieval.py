from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import os
import pickle
import numpy as np
import json


class InformationRetrieval():
	def __init__(self):
		self.index = None
		self.N = None
		self.V = None
		self.docIDs = None
		self.doc_lengths= None
		self.vocabulary = None

	def buildIndex(self, docs, docIDs, preprocess_runtime):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		start_time=time()
		if os.path.exists('Large_files/inverted_index.json') and not preprocess_runtime:
			with open('Large_files/inverted_index.json', 'r') as fp:
				self.index = json.load(fp)
				self.V = len(list(self.index.keys()))
				self.N = len(docIDs) # number of documents in the corpus
				self.docIDs = docIDs

				self.doc_lengths = dict([(docID, len([word for sentence in doc for word in sentence])) for docID, doc in zip(docIDs, docs)])
				print("***Inverted index loaded from memory***")
				end_time = time()
				print("Time taken to load the inverted index from memory: {} seconds".format(end_time-start_time))
				return

		index = {} # dictionary that stores the inverted index
		# they keys are the terms, the values are lists of tuples. Each tuple contains a document ID followed by the term frequency

		doc_lengths = {}
		self.vocabulary = []

		# Each sub-list is a document.
		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			flat_doc = [word for sentence in doc for word in sentence]
			self.vocabulary = self.vocabulary + list(set(flat_doc))
			# Get the frequency of each word in the document
			word_counts = Counter(flat_doc)
			# Unique words in the document.
			unique_words = list(word_counts.keys())

			doc_lengths[docIDs[i]] = len(flat_doc)

			for unique_word in unique_words:
				# when the term exists in the index dictionary
				if unique_word in index:
					index[unique_word].append((docIDs[i], word_counts[unique_word])) # Store the document ID and term frequency in a tuple
				# Create a new key if the term does not exist in the dictionary
				else:
					index[unique_word] = [(docIDs[i], word_counts[unique_word])]

		self.vocabulary = list(set(self.vocabulary))
		if not os.path.exists("Output/vocabulary.txt"):
			with open("Output/vocabulary.txt", "w+") as f:
				for w in self.vocabulary:
					f.write(w+"\n")
			print("Saved vocabulary of preprocessed docs\n")

		self.index = index
		self.V = len(list(index.keys()))
		self.N = len(docIDs) # number of documents in the corpus
		self.docIDs = docIDs
		self.doc_lengths = doc_lengths

		# Save the inverted index after creating
		with open('Large_files/inverted_index.json', 'w+') as fp:
		    json.dump(index, fp)

		print("***Inverted index built and saved at runtime***")
		end_time = time()
		print("Time taken to build and save the inverted index: {} seconds".format(end_time-start_time))


	def autocomplete(self, query, out_folder):
		"""
		Function for enabling auto-completion of queries
		"""
		terms = sorted(list(self.index.keys())) # list of terms
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		# Flatten the list of lists of words into one list of words
		flat_query = [word for sentence in query for word in sentence]
		# Get the frequency of each word in the query
		word_counts = Counter(flat_query)
		# Unique words in the query.
		# unique_words = list(word_counts.keys())
		unique_words = [word for word in word_counts.keys() if word in terms]
		# query_length = len(flat_query)
		# Initialize the query vector as a vector of zeros
		query_vector = np.zeros(self.V)
		for unique_word in unique_words:
			term_index = terms.index(unique_word)
			# TF-IDF scores for the query terms
			query_vector[term_index] = (word_counts[unique_word]) * idf_score[unique_word]

		query_tfidf = np.load('Large_files/query_tfidf.npy')
		ordered_queries = []
		for i in range(len(query_tfidf)):
			# Here, only the dot product of tf-idf vectors is considered. Normalization is not done as varying lengths
			# of queries tend to affect the output more than expected
			sim = np.sum(np.multiply(query_tfidf[i], query_vector)) #/(np.linalg.norm(query_tfidf[i])*np.linalg.norm(query_vector)))
			ordered_queries.append((sim, i+1))

		ordered_queries.sort(reverse=True)
		return ordered_queries


	def rank(self, queries, preprocess, out_folder, custom):
		"""
		Rank the documents according to relevance for each query.
		Ranking is done based on TF-IDF scores.

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][term_index] = (freq/self.doc_lengths[docid]) * idf_score[term]
		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix = {} seconds".format(end_tfidf-start_tfidf))

		query_count = 0
		query_tfidf = np.zeros((len(queries), self.V))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms
				query_vector[term_index] = (word_counts[unique_word]/query_length) * idf_score[unique_word]

			query_tfidf[query_count] = query_vector*query_length # Remove normalization by query length for auto-complete
			query_count +=1

			# Approach 1
			ordered_docs = []

			for key in tfidf_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(tfidf_vectors[key], query_vector))/(np.linalg.norm(tfidf_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		if not custom and (preprocess or not os.path.exists('Large_files/query_tfidf.npy' or len(terms)!=8282)):
			np.save('Large_files/query_tfidf.npy', query_tfidf)
			print("query_tfidf.npy was created")

		if not custom and (preprocess or not os.path.exists('Large_files/glove_arr.npy') or len(terms)!=8282):
			with open('Large_files/glove_cranfield.6B.300d.pickle', 'rb') as f:
				wvec = pickle.load(f)
			glove_arr = np.zeros((len(terms), 300))
			for i in range(len(terms)):
				glove_arr[i] = wvec[terms[i]]
			np.save('Large_files/glove_arr.npy', glove_arr)
			print("Large_files/glove_arr.npy was created")

		return doc_IDs_ordered


	def rank_sentence_wise(self, queries, preprocess, out_folder, custom):
		"""
		Rank the documents according to relevance for each query.
		TF-IDF scores with sentence-wise comparison is used.


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][term_index] = (freq/self.doc_lengths[docid]) * idf_score[term]
		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix = {} seconds".format(end_tfidf-start_tfidf))

		query_count = 0
		query_tfidf = np.zeros((len(queries), self.V))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms
				query_vector[term_index] = (word_counts[unique_word]/query_length) * idf_score[unique_word]

			query_tfidf[query_count] = query_vector*query_length # Remove normalization by query length for auto-complete
			query_count +=1

			# Approach 1
			ordered_docs = []

			for key in tfidf_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(tfidf_vectors[key], query_vector))/(np.linalg.norm(tfidf_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		if not custom and (preprocess or not os.path.exists('Large_files/query_tfidf.npy') or len(terms)!=8282):
			np.save('Large_files/query_tfidf.npy', query_tfidf)
			print("Large_files/query_tfidf.npy was created")

		return doc_IDs_ordered

	def rank_glove_weighted(self, queries):
		"""
		Rank the documents according to relevance for each query.
		TF-IDF vectors with query expansion using GloVe similarity is used.


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		glove_arr = np.load('Large_files/glove_arr.npy')
		for arr_pos in range(len(glove_arr)):
			glove_arr[arr_pos] = glove_arr[arr_pos]/np.linalg.norm(glove_arr[arr_pos])

		glove_sim_arr = np.zeros((glove_arr.shape[0], glove_arr.shape[0]))
		for i in range(len(glove_arr)):
			glove_sim_arr[i] = np.dot(glove_arr, glove_arr[i])

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				vec = (freq/self.doc_lengths[docid]) * idf_score[term] * 0.01 * glove_sim_arr[term_index]
				vec[vec<0.5] = 0
				tfidf_vectors[docid] += vec
				tfidf_vectors[docid][term_index] += (freq/self.doc_lengths[docid]) * idf_score[term]
		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix = {} seconds".format(end_tfidf-start_tfidf))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms with query expansion using GloVe similarity
				vec = (word_counts[unique_word]/query_length) * idf_score[unique_word] * 0.01 * glove_sim_arr[term_index]
				vec[vec<0.5] = 0
				query_vector += vec
				query_vector[term_index] += (word_counts[unique_word]/query_length) * idf_score[unique_word]

			# Approach 1
			ordered_docs = []

			for key in tfidf_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(tfidf_vectors[key], query_vector))/(np.linalg.norm(tfidf_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def rank_LSA(self, queries, k, preprocess):
		"""
		Rank the documents according to relevance for each query using LSA.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		arg2 : k, i.e., number of components for performing SVD
		arg3 : argument that states whether preprocessing is done at runtime or not

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][term_index] = (freq/self.doc_lengths[docid]) * idf_score[term]
				# If we want to use the term document matrix for LSA
				# tfidf_vectors[docid][term_index] = freq/self.doc_lengths[docid]
				# If we want to use the Term document matrix without normalizing
				# tfidf_vectors[docid][term_index] = freq
		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix in LSA = {} seconds".format(end_tfidf-start_tfidf))

		######### LSA
		# We have a dictionary of tf-idf vectors, if we arrange them column-wise, we get the Term-Document matrix
		k = int(k)
		print("Number of latent dimensions used = {}".format(k))
		keys, vectors = zip(*tfidf_vectors.items())
		# Use the saved SVD results if they are available:
		if os.path.exists("Large_files/SVD.npz") and not preprocess:
			start_svd = time()
			data_dict = np.load("Large_files/SVD.npz")
			print("Found SVD results stored in the folder.")
			U = data_dict["arr_0"]
			S = data_dict["arr_1"]
			Vt = data_dict["arr_2"]
			end_svd = time()
			print("Time taken to load SVD results = {} seconds".format(end_svd - start_svd))
		else:
			print("Calculating SVD at run-time.")
			all_vectors_matrix = np.array(vectors).T
			# Singular value decomposition
			start_svd = time()
			U, S, Vt = np.linalg.svd(all_vectors_matrix)
			end_svd = time()
			print("Time taken to do SVD = {} seconds".format(end_svd - start_svd))
			np.savez_compressed("Large_files/SVD.npz", U, S,Vt)# Save the results

		Uk = U[:, :k]
		# np.save("Large_files/Singular_values.npy", S)
		Sk = np.diag(S[:k])
		# Dictionary of document vectors in the latent space (k dimensions)
		lsa_doc_vectors = dict(zip(keys, Vt[:k, :].T))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms
				query_vector[term_index] = (word_counts[unique_word]/query_length) * idf_score[unique_word]

			# Approach 1
			ordered_docs = []
			# Map the query vector to the latent space
			query_vector = (query_vector.reshape(1, self.V) @ Uk) @ np.linalg.inv(Sk)

			for key in lsa_doc_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(lsa_doc_vectors[key], query_vector))/(np.linalg.norm(lsa_doc_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def rank_LSA_glove_weighted(self, queries, k, preprocess):
		"""
		Rank the documents according to relevance for each query using LSA
		using vectors obtained after query expansion with GloVe similarity.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		glove_arr = np.load('Large_files/glove_arr.npy')
		for arr_pos in range(len(glove_arr)):
			glove_arr[arr_pos] = glove_arr[arr_pos]/np.linalg.norm(glove_arr[arr_pos])

		glove_sim_arr = np.zeros((glove_arr.shape[0], glove_arr.shape[0]))
		for i in range(len(glove_arr)):
			glove_sim_arr[i] = np.dot(glove_arr, glove_arr[i])

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][term_index] += (freq/self.doc_lengths[docid]) * idf_score[term]
				tfidf_vectors[docid] += (freq/self.doc_lengths[docid]) * idf_score[term] * 0.01 * glove_sim_arr[term_index]

		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix in LSA = {} seconds".format(end_tfidf-start_tfidf))

		######### LSA
		# We have a dictionary of tf-idf vectors, if we arrange them column-wise, we get the Term-Document matrix
		k = int(k)
		print("Number of latent dimensions used = {}".format(k))
		keys, vectors = zip(*tfidf_vectors.items())
		# Use the saved SVD results if they are available:
		if os.path.exists("Large_files/SVD_glove_weighted.npz") and not preprocess:
			start_svd = time()
			data_dict = np.load("Large_files/SVD_glove_weighted.npz")
			print("Found SVD_glove_weighted results stored in the folder.")
			U = data_dict["arr_0"]
			S = data_dict["arr_1"]
			Vt = data_dict["arr_2"]
			end_svd = time()
			print("Time taken to load SVD results = {} seconds".format(end_svd - start_svd))
		else:
			print("Calculating SVD at run-time.")
			all_vectors_matrix = np.array(vectors).T
			# Singular value decomposition
			start_svd = time()
			U, S, Vt = np.linalg.svd(all_vectors_matrix)
			end_svd = time()
			print("Time taken to do SVD = {} seconds".format(end_svd - start_svd))
			np.savez_compressed("Large_files/SVD_glove_weighted.npz", U, S,Vt)# Save the results

		Uk = U[:, :k]
		# np.save("Large_files/Singular_values.npy", S)
		Sk = np.diag(S[:k])
		# Dictionary of document vectors in the latent space (k dimensions)
		lsa_doc_vectors = dict(zip(keys, Vt[:k, :].T))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms
				query_vector[term_index] += (word_counts[unique_word]/query_length) * idf_score[unique_word]
				query_vector += (word_counts[unique_word]/query_length) * idf_score[unique_word] * 0.01 * glove_sim_arr[term_index]

			# Approach 1
			ordered_docs = []
			# Map the query vector to the latent space
			query_vector = (query_vector.reshape(1, self.V) @ Uk) @ np.linalg.inv(Sk)

			for key in lsa_doc_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(lsa_doc_vectors[key], query_vector))/(np.linalg.norm(lsa_doc_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def rank_LSA_wordnet_weighted(self, queries, k, preprocess):
		"""
		Rank the documents according to relevance for each query using LSA
		using vectors obtained after query expansion with WordNet Wu-Palmer similarity.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		start_tfidf = time()
		# Calculating IDF for every term in the corpus
		terms = sorted(list(self.index.keys())) # list of terms

		# IDF score dictionary
		idf_score = {}
		for term in terms:
			# Number of docs in which the term occurs
			n = len(self.index[term])
			idf_score[term] = np.log10(self.N/n)

		tfidf_vectors = {}
		for docid in self.docIDs:
			if self.doc_lengths[docid]>0:
				# Initialize each document vector as a vector of zeroes
				tfidf_vectors[docid] = np.zeros(self.V)

		wordnet_sim_arr = np.load("Large_files/wordnet_sim_arr.npy")

		for term_index, term in enumerate(terms):
			for pair in self.index[term]:
				docid = pair[0]
				freq = pair[1]
				# Calculating Tf*IDF
				tfidf_vectors[docid][term_index] += (freq/self.doc_lengths[docid]) * idf_score[term]
				tfidf_vectors[docid] += (freq/self.doc_lengths[docid]) * idf_score[term] * 0.03 * wordnet_sim_arr[term_index]

		end_tfidf = time()
		print("Time taken to build the TF-IDF matrix in LSA = {} seconds".format(end_tfidf-start_tfidf))

		######### LSA
		# We have a dictionary of tf-idf vectors, if we arrange them column-wise, we get the Term-Document matrix
		k = int(k)
		print("Number of latent dimensions used = {}".format(k))
		keys, vectors = zip(*tfidf_vectors.items())
		# Use the saved SVD results if they are available:
		if os.path.exists("Large_files/SVD_wordnet_weighted.npz") and not preprocess:
			start_svd = time()
			data_dict = np.load("Large_files/SVD_wordnet_weighted.npz")
			print("Found SVD_wordnet_weighted results stored in the folder.")
			U = data_dict["arr_0"]
			S = data_dict["arr_1"]
			Vt = data_dict["arr_2"]
			end_svd = time()
			print("Time taken to load SVD results = {} seconds".format(end_svd - start_svd))
		else:
			print("Calculating SVD at run-time.")
			all_vectors_matrix = np.array(vectors).T
			# Singular value decomposition
			start_svd = time()
			U, S, Vt = np.linalg.svd(all_vectors_matrix)
			end_svd = time()
			print("Time taken to do SVD = {} seconds".format(end_svd - start_svd))
			np.savez_compressed("Large_files/SVD_wordnet_weighted.npz", U, S,Vt)# Save the results

		Uk = U[:, :k]
		# np.save("Large_files/Singular_values.npy", S)
		Sk = np.diag(S[:k])
		# Dictionary of document vectors in the latent space (k dimensions)
		lsa_doc_vectors = dict(zip(keys, Vt[:k, :].T))

		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			# Get the frequency of each word in the query
			word_counts = Counter(flat_query)
			# Unique words in the query.
			# unique_words = list(word_counts.keys())
			unique_words = [word for word in word_counts.keys() if word in terms]
			# Find the length of the query
			query_length = len(flat_query)
			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(self.V)
			for unique_word in unique_words:
				term_index = terms.index(unique_word)
				# TF-IDF scores for the query terms
				query_vector[term_index] += (word_counts[unique_word]/query_length) * idf_score[unique_word]
				query_vector += (word_counts[unique_word]/query_length) * idf_score[unique_word] * 0.03 * wordnet_sim_arr[term_index]

			# Approach 1
			ordered_docs = []
			# Map the query vector to the latent space
			query_vector = (query_vector.reshape(1, self.V) @ Uk) @ np.linalg.inv(Sk)

			for key in lsa_doc_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(lsa_doc_vectors[key], query_vector))/(np.linalg.norm(lsa_doc_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def train_glove(self, all_docs):
		"""
		Function for getting Cranfield-specific GloVe vectors
		"""

		import gensim
		from gensim.models import Word2Vec

		with open('Large_files/glove.6B.300d.pickle', 'rb') as f:
			wvec = pickle.load(f)

		old_len = len(wvec)

		em_model = Word2Vec(all_docs, vector_size=300, window=5, min_count=1, workers=2)
		word_embedding_dict = {word: vec for word, vec in zip(em_model.wv.index_to_key, em_model.wv.vectors)}

		# Updating GloVe with the new vectors
		for word in list(word_embedding_dict.keys()):
			if word in wvec:
				wvec.update({word : 0.9*word_embedding_dict[word]+0.1*wvec[word]})
			else:
				wvec.update({word : word_embedding_dict[word]})

		print("Number of new words added to GloVe vocabulary =", len(list(wvec.keys()))-old_len)

		with open('Large_files/glove_cranfield.6B.300d.pickle', 'wb') as f:
			pickle.dump(wvec, f)
		return

	def rank_glove(self, docs, docIDs, queries, k):
		"""
		Rank the documents according to relevance for each query using GloVe representions

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# Read GloVe vectors file
		# glove_dim = 300

		# with open('Large_files/glove.6B.300d.pickle', 'rb') as f:
		# 	glove_dict = pickle.load(f)

		with open('Large_files/glove_cranfield.6B.300d.pickle', 'rb') as f:
			glove_dict = pickle.load(f)

		doc_IDs_ordered = []

		glove_docs = {} # np.zeros((len(docs), glove_dim))

		all_docs = []

		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			flat_doc = [word for sentence in doc for word in sentence]
			all_docs.append(flat_doc)
			if len(flat_doc)==0:
				continue
			glove_docs[docIDs[i]] = np.sum(np.array([glove_dict[word] for word in flat_doc if word in glove_dict]), axis=0)

			# Give 3 times the weighting to the title
			flat_sent = [word for word in doc[0]]
			glove_docs[docIDs[i]] += 2*np.sum(np.array([glove_dict[word] for word in flat_sent if word in glove_dict]), axis=0)

		if not os.path.exists('Large_files/glove_cranfield.6B.300d.pickle'):
			self.train_glove(all_docs)
			print("Trained GloVe vectors have been saved!")
		del all_docs

		for query in queries:
			ordered_docs = []
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			glove_query = np.sum(np.array([glove_dict[word] for word in flat_query if word in glove_dict]), axis=0)
			for key in glove_docs:
				sim = np.sum(np.multiply(glove_docs[key], glove_query))/(np.linalg.norm(glove_docs[key])*np.linalg.norm(glove_query))
				ordered_docs.append((sim, key))

			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def rank_glove_sentence_wise(self, docs, docIDs, queries, k):
		"""
		Rank the documents according to relevance for each query using GloVe representions.
		Comparison is made between the query and each sentence at a time from each document

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query


		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		# Read GloVe vectors file
		# glove_dim = 300

		# with open('Large_files/glove.6B.300d.pickle', 'rb') as f:
		# 	glove_dict = pickle.load(f)

		with open('Large_files/glove_cranfield.6B.300d.pickle', 'rb') as f:
			glove_dict = pickle.load(f)

		doc_IDs_ordered = []

		glove_docs = {} # np.zeros((len(docs), glove_dim))

		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			if len(doc)==0:
				continue
			flat_sent = [word for word in doc[0]]
			title_glove = np.sum(np.array([glove_dict[word] for word in flat_sent if word in glove_dict]), axis=0)

			flat_doc = []
			for sentence in doc:
				flat_doc.append([word for word in sentence])
			if len(flat_doc)==0:
				continue
			glove_docs[docIDs[i]] = [title_glove+np.sum(np.array([glove_dict[word] for word in sent if word in glove_dict]), axis=0) for sent in flat_doc]

		for query in queries:
			ordered_docs = []
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]
			glove_query = np.sum(np.array([glove_dict[word] for word in flat_query if word in glove_dict]), axis=0)
			for key in glove_docs:
				sim_list=[]
				for k in range(len(glove_docs[key])):
					if np.linalg.norm(glove_docs[key][k]) == 0:
						continue
					sim_list.append(np.sum(np.multiply(glove_docs[key][k], glove_query))/(np.linalg.norm(glove_docs[key][k])*np.linalg.norm(glove_query)))
				ordered_docs.append((np.max(sim_list), key))

			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered

	def rank_univ_sentence_enc(self, docs, docIDs, queries, k):
		"""
		Function that uses Universal Sentence encoder to rank documents
		"""

		import tensorflow as tf
		import tensorflow_hub as hub
		from packaging import version
		if not version.parse(tf.__version__) >= version.parse('2'):
			print("The Universal Sentence Encoder requires tensorflow version > 2. \
				Please upgrade to the latest tf version.")

		# import tarfile
		# my_tar = tarfile.open('Large_files/4?tf-hub-format=compressed')
		# my_tar.extractall('Large_files/') # specify which folder to extract to
		# my_tar.close()

		module_url = "Large_files/"
		model = hub.load(module_url)
		print("USE model loaded successfully!")

		doc_IDs_ordered = []
		USE_docs = {}

		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			if len(doc)==0:
				continue
			flat_doc = [' '.join([word for sentence in doc for word in sentence])]
			flat_sent = [' '.join([word for word in doc[0]])]
			USE_docs[docIDs[i]] = np.array(model(flat_doc)[0])+ 2*np.array(model(flat_sent)[0])

		for query in queries:
			ordered_docs = []
			# Flatten the list of lists of words into one list of words
			flat_query = [' '.join([word for sentence in query for word in sentence])]
			USE_query = np.array(model(flat_query)[0])
			# print(USE_query.shape)
			for key in USE_docs:
				sim = np.sum(np.multiply(USE_docs[key], USE_query))/(np.linalg.norm(USE_docs[key])*np.linalg.norm(USE_query))
				ordered_docs.append((sim, key))

			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered


	def rank_univ_sentence_enc_sentence_wise(self, docs, docIDs, queries, k):
		"""
		Function that uses Universal Sentence encoder to rank documents.
		Sentence-wise comparison is used.
		"""

		import tensorflow as tf
		import tensorflow_hub as hub
		from packaging import version
		if not version.parse(tf.__version__) >= version.parse('2'):
			print("The Universal Sentence Encoder requires tensorflow version > 2. \
				Please upgrade to the latest tf version.")

		# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
		# model = hub.load(module_url)
		# print ("module %s loaded" % module_url)

		module_url = "Large_files/"
		model = hub.load(module_url)
		print("USE model loaded successfully!")

		doc_IDs_ordered = []
		USE_docs = {}

		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			if len(doc)==0:
				continue
			flat_doc = [[' '.join([word for word in sentence])] for sentence in doc]
			flat_sent = [' '.join([word for word in doc[0]])]
			# Give thrice weight to title by adding twice the title to the document
			USE_docs[docIDs[i]] = [np.array(model(flat_doc[sent])[0])+ 2*np.array(model(flat_sent)[0]) for sent in range(len(flat_doc))]

		for query in queries:
			ordered_docs = []
			# Flatten the list of lists of words into one list of words
			flat_query = [' '.join([word for sentence in query for word in sentence])]
			USE_query = np.array(model(flat_query)[0])
			for key in USE_docs:
				sim_list=[]
				for k in range(len(USE_docs[key])):
					if np.linalg.norm(USE_docs[key][k]) == 0:
						continue
					sim_list.append(np.sum(np.multiply(USE_docs[key][k], USE_query))/(np.linalg.norm(USE_docs[key][k])*np.linalg.norm(USE_query)))
				ordered_docs.append((np.max(sim_list), key))

			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered

	def rank_ESA(self, queries, docs, docIDs):
		"""
		Rank the documents according to relevance for each query using ESA

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""
		# Load the saved vectors
		with open("Output/esa_vectors.json", "r") as f:
			esa_vectors = json.load(f)

		with open('Output/article_dict.json', 'r') as fp:
			article_dict = json.load(fp)
		pageid_list = list(article_dict.keys())

		esa_doc_lengths = {}
		for pid in pageid_list:
			esa_doc_lengths[pid] = sum(article_dict[pid].values())

		# Dictionary of document vectors
		esa_doc_vectors = {}
		doc_IDs_ordered= []

		# Each sub-list is a document.
		for i, doc in enumerate(docs):
			# Flatten the list of lists of words into one list of words
			flat_doc = [word for sentence in doc for word in sentence]

			esa_doc_vectors[docIDs[i]] = np.zeros(len(list(article_dict.keys())), dtype="float")
			for word in flat_doc:
				esa_doc_vectors[docIDs[i]] = esa_doc_vectors[docIDs[i]] + np.array(esa_vectors[word])


		# Each sub-list is a query.
		for query in queries:
			# Flatten the list of lists of words into one list of words
			flat_query = [word for sentence in query for word in sentence]

			# Initialize the query vector as a vector of zeros
			query_vector = np.zeros(len(pageid_list), dtype="float")

			for word in flat_query:
				if word in esa_vectors:
					query_vector = query_vector + np.array(esa_vectors[word])
				else:
					word_vector = np.zeros(len(pageid_list), dtype="float")
					for j, pid in enumerate(pageid_list):
						if word in article_dict[pid]:
							word_vector[j] = article_dict[pid][word]/esa_doc_lengths[pid]

					query_vector = query_vector + word_vector

			ordered_docs = []

			for key in esa_doc_vectors:
				# Calculate cosine similarity between each document and the query
				sim = np.sum(np.multiply(esa_doc_vectors[key], query_vector))/(np.linalg.norm(esa_doc_vectors[key])*np.linalg.norm(query_vector))
				ordered_docs.append((sim, key))

			# Sorting the documents based on the cosine_similarity
			ordered_docs.sort(reverse=True)
			doc_IDs_ordered.append([pair[1] for pair in ordered_docs])

		return doc_IDs_ordered
