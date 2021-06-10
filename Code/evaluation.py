import numpy as np


class Evaluation():

	def __init__(self):
		self.true_docs_relevance = None

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		returned_docs = query_doc_IDs_ordered[:k] # take the first k returned docs
		# print(returned_docs)
		# List of relevant docs among the top k returned docs
		relevant_among_top_k_returned = [doc for doc in returned_docs if doc in true_doc_IDs]

		# To avoid 0 in the denominator when calculating precision
		if len(returned_docs) == 0:
			return 1

		# print(len(relevant_among_top_k_returned), len(returned_docs))
		# Precision = |Returned & Relevant|/|Returned|
		precision = len(relevant_among_top_k_returned)/len(returned_docs)

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		meanPrecision = 0
		for i, qid in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i] # IR system result for this query
			relevant_docs = [(d["position"], int(d["id"])) for d in qrels if d["query_num"]==str(qid)] # Select the documents relevant to this query
			relevant_docs.sort() # sort the results in ascending order of "position"

			true_doc_IDs = [pair[1] for pair in relevant_docs] # Extract only the document id from each tuple

			# Call the queryPrecision function to get precision @ k measure for this query
			meanPrecision += self.queryPrecision(query_doc_IDs_ordered, qid, true_doc_IDs, k)

		meanPrecision = meanPrecision/len(query_ids) # mean of precision values over all the queries

		return meanPrecision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		returned_docs = query_doc_IDs_ordered[:k] # take the first k returned docs

		# List of relevant docs among the top k returned docs
		relevant_among_top_k_returned = [doc for doc in returned_docs if doc in true_doc_IDs]

		# To avoid 0 in the denominator when calculating recall
		if len(true_doc_IDs)==0:
			return 1

		# Recall = |Returned & Relevant|/|Relevant|
		recall = len(relevant_among_top_k_returned)/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0

		for i, qid in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i] # IR system result for this query
			relevant_docs = [(d["position"], int(d["id"])) for d in qrels if d["query_num"]==str(qid)] # Select the documents relevant to this query
			relevant_docs.sort() # sort the results in ascending order of "position"

			true_doc_IDs = [pair[1] for pair in relevant_docs] # Extract only the document id from each tuple

			# Call the queryRecall function to get precision @ k measure for this query
			meanRecall += self.queryRecall(query_doc_IDs_ordered, qid, true_doc_IDs, k)

		meanRecall = meanRecall/len(query_ids) # mean of recall values over all the queries

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		# Find precision and recall values for the query
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		# If no returned documents are relevant:
		if precision==0 and recall==0:
			return 0
		# Computing F-score = Harmonic Mean(precision, recall)
		fscore = 2 * precision * recall / (precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0

		for i, qid in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i] # IR system result for this query
			relevant_docs = [(d["position"], int(d["id"])) for d in qrels if d["query_num"]==str(qid)] # Select the documents relevant to this query
			relevant_docs.sort() # sort the results in ascending order of "position"

			true_doc_IDs = [pair[1] for pair in relevant_docs] # Extract only the document id from each tuple

			# Call the queryFscore function to get precision @ k measure for this query
			meanFscore += self.queryFscore(query_doc_IDs_ordered, qid, true_doc_IDs, k)

		meanFscore = meanFscore/len(query_ids) # mean of F-score values over all the queries

		return meanFscore


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		DCG = 0
		IDCG = 0

		returned_docs = query_doc_IDs_ordered[:k] # take the first k returned docs

		# Get the degree of relevance of each relevant document for the query
		# position 1 has highest degree of relevance and position 4 has lowest.
		# max_position = true_doc_IDs[-1][0]
		# true_docs_relevance = [5-pair[0] for pair in true_doc_IDs]
		# true_docs = [pair[1] for pair in true_doc_IDs]

		found_relevances = [] # To store observed relevance values

		for i in range(1, k+1):
			# Find degree of relevance of each retrieved doc
			if returned_docs[i-1] in true_doc_IDs:
				doc_relevance = self.true_docs_relevance[true_doc_IDs.index(returned_docs[i-1])]
			# If the document is not found in the set of relevant docs, set the relevance to 0.
			else:
				doc_relevance = 0
			DCG += (doc_relevance/np.log2(i+1)) # Compute DCG

		# Compute ideal DCG by ordering relevant in the ideal order
		found_relevances = self.true_docs_relevance.copy()
		# If the number of relevant documents is less than k, append 0's to signify 0 relevance
		if len(found_relevances)<k:
			found_relevances.extend([0]*(k-len(found_relevances)))
		IDCG = np.sum([found_relevances[i-1]/np.log2(i+1) for i in range(1, k+1)]) # Computing IDCG

		# If there are no relevant documents:
		if IDCG == 0:
			return 0

		nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0

		for i, qid in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i] # IR system result for this query
			relevant_docs = [(d["position"], int(d["id"])) for d in qrels if d["query_num"]==str(qid)] # Select the documents relevant to this query
			relevant_docs.sort() # sort the results in ascending order of "position"

			if len(relevant_docs)==0:
				print("Found 0 len rel doc at query ", qid)

			# Store relevance of the relevant documents. As per the README file in Cranfield dataset, as the
			# position number increases the relevance decreases. Hence, relevance is computed as 5 - position
			# where position is based on the position number given in cran_qrels.json.
			true_docs_relevance = [5-pair[0] for pair in relevant_docs]
			self.true_docs_relevance = true_docs_relevance
			true_doc_IDs = [pair[1] for pair in relevant_docs] # Extract only the document id from each tuple

			# Call the queryNDCG function to get precision @ k measure for this query
			meanNDCG += self.queryNDCG(query_doc_IDs_ordered, qid, true_doc_IDs, k)

		meanNDCG = meanNDCG/len(query_ids) # mean of nDCG values over all the queries

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0
		returned_and_relevant_positions = []

		# Finding positions in the ordering of documents by our system where a relevant document is returned
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				returned_and_relevant_positions.append(i)

		# If there are no relevant documents returned:
		if len(returned_and_relevant_positions)==0:
			return 0

		# Sum the precision values across positions where a relevant document was returned
		for i in returned_and_relevant_positions:
			avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1)

		# Divide by the number of relevant documents returned up to that position to get average precision (AP)
		avgPrecision = avgPrecision/len(returned_and_relevant_positions)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0

		for i, qid in enumerate(query_ids):
			query_doc_IDs_ordered = doc_IDs_ordered[i] # IR system result for this query
			relevant_docs = [(d["position"], int(d["id"])) for d in q_rels if d["query_num"]==str(qid)] # Select the documents relevant to this query
			relevant_docs.sort() # sort the results in ascending order of "position"

			true_doc_IDs = [pair[1] for pair in relevant_docs] # Extract only the document id from each tuple

			# Call the queryAveragePrecision function to get AP @ k measure for this query
			meanAveragePrecision += self.queryAveragePrecision(query_doc_IDs_ordered, qid, true_doc_IDs, k)

		meanAveragePrecision = meanAveragePrecision/len(query_ids) # mean of AP values over all the queries

		return meanAveragePrecision
