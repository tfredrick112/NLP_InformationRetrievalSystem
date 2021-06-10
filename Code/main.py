from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from spellCheck import *
from bigramSpellCheck import BigramSpellCheck
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from time import time
import multiprocessing
from joblib import Parallel, delayed

from sys import version_info
import argparse
import os
import json
import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # This line can be deleted if it gives an error
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.informationRetriever = InformationRetrieval()
        self.evaluator = Evaluation()


    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
        Call the required stemmer/lemmatizer
        """
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
        Call the required stopword remover
        """
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        """
        Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
        """

        # Number of CPU cores
        num_cores = multiprocessing.cpu_count()

        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        if not self.args.custom:
            json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        # else:
        #     json.dump(segmentedQueries, open(self.args.out_folder + "custom_segmented_queries.txt", 'w'))

        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        if not self.args.custom:
            json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        # else:
        #     json.dump(tokenizedQueries, open(self.args.out_folder + "custom_tokenized_queries.txt", 'w'))

        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        if not self.args.custom:
            json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        # else:
        #     json.dump(reducedQueries, open(self.args.out_folder + "custom_reduced_queries.txt", 'w'))

        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        if not self.args.custom:
            json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))
        # else:
        #     json.dump(stopwordRemovedQueries, open(self.args.out_folder + "custom_stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        """
        Preprocess the documents
        """
        # Number of CPU cores
        num_cores = multiprocessing.cpu_count()

        # Segment docs
        start = time()
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)

        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        end = time()
        print("Time taken to segment sentences = {} s".format(end-start))

        # Tokenize docs
        start = time()
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)

        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        end = time()
        print("Time taken to tokenize = {} s".format(end-start))

        # Stem/Lemmatize docs
        start = time()
        # reducedDocs = []

        try:
            reducedDocs = [self.reduceInflection(doc) for doc in tokenizedDocs]
        except: # Serialization using joblibs gives errors in colab (python version issue)
            reducedDocs = Parallel(n_jobs = num_cores)(delayed(self.reduceInflection)(doc) for doc in tokenizedDocs)

        # for doc in tokenizedDocs:
        #     reducedDoc = self.reduceInflection(doc)
        #     reducedDocs.append(reducedDoc)

        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        end = time()
        print("Time taken to do inflection reduction = {} s".format(end-start))


        # Remove stopwords from docs
        start = time()
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
        end = time()
        print("Time taken to remove stopwords = {} s".format(end-start))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs


    def evaluateDataset(self):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP
          for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """
        #### Read queries
        # If you want to do the preprocessing ar run time.
        queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], [item["query"] for item in queries_json]
        if self.args.preprocess or (not os.path.exists(self.args.out_folder + "stopword_removed_queries.txt")):
            # Process queries
            start_time=time()
            processedQueries = self.preprocessQueries(queries)
            end_time = time()
            print("Time taken to preprocess queries: {} seconds".format(end_time-start_time))
            print('----------------------------------------------------------')
        else:
            # In this option, we read the preprocessed queries directly
            processedQueries = json.load(open(self.args.out_folder + "stopword_removed_queries.txt", 'r'))[:]

        ##### Read documents
        # This choice is selected if preprocess was specified or if the preprocessed documents cannot be found
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
        if self.args.preprocess or (not os.path.exists(self.args.out_folder + "stopword_removed_docs.txt")):
            # Process documents
            start_time=time()
            processedDocs = self.preprocessDocs(docs)
            end_time = time()
            print("Time taken to preprocess documents: {} seconds".format(end_time-start_time))
            print('----------------------------------------------------------')
        else:
            # In this option, we read the preprocessed docs directly
            processedDocs = json.load(open(self.args.out_folder + "stopword_removed_docs.txt", 'r'))[:]

        #### Building the inverted index
        self.informationRetriever.buildIndex(processedDocs, doc_ids, self.args.preprocess)
        print('----------------------------------------------------------')

        # Rank the documents for each query
        if self.args.model=="vector_space":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank(processedQueries, self.args.preprocess, self.args.out_folder, self.args.custom)
            end_time = time()
            print("Time taken to rank the documents using the Vector space model: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove_weighted_tfidf":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove_weighted(processedQueries)
            end_time = time()
            print("Time taken to rank the documents using the Vector space model: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA(processedQueries, self.args.k, self.args.preprocess)
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa_glove_weighted":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA_glove_weighted(processedQueries, self.args.k, self.args.preprocess)
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa_wordnet_weighted":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA_wordnet_weighted(processedQueries, self.args.k, self.args.preprocess)
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove(processedDocs, doc_ids, processedQueries, self.args.k)
            end_time = time()
            print("Time taken to rank the documents using GloVe: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove_sentence_wise":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove_sentence_wise(processedDocs, doc_ids, processedQueries, self.args.k)
            end_time = time()
            print("Time taken to rank the documents using GloVe: {} seconds".format(end_time-start_time))
        elif self.args.model=="univ_sentence_enc":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_univ_sentence_enc(processedDocs, doc_ids, processedQueries, self.args.k)
            end_time = time()
            print("Time taken to rank the documents using Universal Sentence Encoder: {} seconds".format(end_time-start_time))
        elif self.args.model=="esa":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_ESA(processedQueries, processedDocs, doc_ids)
            end_time = time()
            print("Time taken to rank the documents using ESA: {} seconds".format(end_time-start_time))



        print('----------------------------------------------------------')
        # Read relevance judements
        qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]
        ev_start = time()
        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " + str(k) + " : " + str(precision) + ", " + str(recall) + ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +str(k) + " : " + str(MAP) + ", " + str(nDCG))

        ev_end = time()
        print("Time taken to evaluate = {} seconds".format(ev_end - ev_start))

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.savefig(self.args.out_folder + "eval_plot.png")


    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        # if self.args.interactive:
        #     query = self.args.query
        # else:
        #     query = input()
        ##### Read documents
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
        if self.args.preprocess or (not os.path.exists(self.args.out_folder + "stopword_removed_docs.txt")):
            # Process documents
            start_time=time()
            processedDocs = self.preprocessDocs(docs)
            end_time = time()
            print("Time taken to preprocess documents: {} seconds".format(end_time-start_time))
        else:
            # In this option, we read the preprocessed docs directly
            processedDocs = json.load(open(self.args.out_folder + "stopword_removed_docs.txt", 'r'))[:]


        #Get query
        query = input("Enter your query here: ")

        ### SPELL CHECK
        # Method 1: Unigrams, correct one word at a time
        start_spell = time()
        self.spellcheck = BigramSpellCheck(docs)
        query_corrected = self.spellcheck.correct_query(query)
        # query_corrected = doSpellCheck(query, docs)
        end_spell = time()
        print("Time taken for spell check = {} seconds.".format(end_spell - start_spell))

        # Ask the user about spell-check suggestions only if query_corrected is different from the original
        if query_corrected != query:
            response = input("Did you mean: {}? (y/n/enter a fresh query)\n".format(query_corrected))
            if response=="y":
                # Correct the query if the user responds with "y"
                query = query_corrected
            elif response!="n":
                query = response

        print('----------------------------------------------------------')

        # Process query (after you have done spell check)
        processedQuery = self.preprocessQueries([query])[0]

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids, self.args.preprocess)

        # Generating autocomplete suggestions
        queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
        queries = [item["query"] for item in queries_json]
        ordered_queries = self.informationRetriever.autocomplete(processedQuery, self.args.out_folder)
        ordered_queries = ordered_queries[:5]
        query_log = np.load(self.args.out_folder+"query_log.npy")
        popularity = [[query_log[ordered_query[1], 0], ordered_query[1]] for ordered_query in ordered_queries]
        recentness = [[query_log[ordered_query[1], 1], ordered_query[1]] for ordered_query in ordered_queries]

        sim_max = np.max([pair[0] for pair in ordered_queries])
        pop_max = np.max([pair[0] for pair in popularity])
        rec_max = np.max([pair[0] for pair in recentness])

        print('----------------------------------------------------------')

        if sim_max==0: # If no query contains a word from the user-entered incomplete query
            print("No matching queries found for autocompletion!")
            autocomplete_choice = 0
        else:
            query_weighted_rank = {ordered_queries[i][1]:0.5*ordered_queries[i][0]/sim_max for i in range(len(ordered_queries))}
            for i in range(5):
                query_weighted_rank[popularity[i][1]] += 0.3*(popularity[i][0]/pop_max)
                query_weighted_rank[recentness[i][1]] += 0.2*(recentness[i][0]/rec_max)
            query_weighted_rank = [[q_rank, q_id] for q_id, q_rank in query_weighted_rank.items()]
            query_weighted_rank.sort(reverse=True)
            ordered_queries = [pair[1] for pair in query_weighted_rank]

            print("\nCandidate queries for autocompletion:\n")
            print(0, query, '(Your original query)', '\n')
            for i in range(5):
                print(i+1, queries[ordered_queries[i]-1], "(Query ID: "+str(ordered_queries[i])+")", '\n')
            autocomplete_choice = int(input("Enter a query number for autocompletion. Hit 0 to proceed with your original query: "))

        if autocomplete_choice!=0:
            processedQuery = self.preprocessQueries([queries[ordered_queries[autocomplete_choice-1]-1]])[0]
            query_log[ordered_queries[autocomplete_choice-1],0]+=1
            query_log[ordered_queries[autocomplete_choice-1],1]=time()
            np.save(self.args.out_folder+"query_log.npy", query_log)

        print('----------------------------------------------------------')

        # Rank the documents for the query
        if self.args.model=="vector_space":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank([processedQuery], self.args.preprocess, self.args.out_folder, self.args.custom)[0]
            end_time = time()
            print("Time taken to rank the documents using the Vector space model: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove_weighted_tfidf":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove_weighted([processedQuery])[0]
            end_time = time()
            print("Time taken to rank the documents: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA([processedQuery], self.args.k, self.args.preprocess)[0]
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa_glove_weighted":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA_glove_weighted([processedQuery], self.args.k, self.args.preprocess)[0]
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="lsa_wordnet_weighted":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_LSA_wordnet_weighted([processedQuery], self.args.k, self.args.preprocess)[0]
            end_time = time()
            print("Time taken to rank the documents using LSA: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove(processedDocs, doc_ids, [processedQuery], self.args.k)[0]
            end_time = time()
            print("Time taken to rank the documents using GloVe: {} seconds".format(end_time-start_time))
        elif self.args.model=="glove_sentence_wise":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_glove_sentence_wise(processedDocs, doc_ids, [processedQuery], self.args.k)[0]
            end_time = time()
            print("Time taken to rank the documents using GloVe: {} seconds".format(end_time-start_time))
        elif self.args.model=="univ_sentence_enc":
            # self.informationRetriever.train_glove(processedDocs)
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_univ_sentence_enc(processedDocs, doc_ids, [processedQuery], self.args.k)[0]
            end_time = time()
            print("Time taken to rank the documents using Universal Sentence Encoder: {} seconds".format(end_time-start_time))
        elif self.args.model=="esa":
            start_time = time()
            doc_IDs_ordered = self.informationRetriever.rank_ESA([processedQuery], processedDocs, doc_ids)[0]
            end_time = time()
            print("Time taken to rank the documents using ESA: {} seconds".format(end_time-start_time))


        print('----------------------------------------------------------')
        # Print the IDs of first five documents
        print("\nTop five documents retrieved : \n")
        for id_ in doc_IDs_ordered[:5]:
            print(str(id_)+":", docs[id_-1], "\n")

        with open("Output/custom_query_results.txt", "w") as f:
            f.write("Query: {}\n".format(query))
            for id_ in doc_IDs_ordered[:5]:
                f.write(str(id_)+":" + docs[id_-1]+"\n")



    def evaluate_autocomplete(self):
        """
        Evaluate the autocomplete feature by using incomplete queries generated from parse trees of Cranfield queries
        """

        ##### Read documents
        import pickle
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
        if self.args.preprocess or (not os.path.exists(self.args.out_folder + "stopword_removed_docs.txt")):
            # Process documents
            start_time=time()
            processedDocs = self.preprocessDocs(docs)
            end_time = time()
            print("Time taken to preprocess documents: {} seconds".format(end_time-start_time))
        else:
            # In this option, we read the preprocessed docs directly
            processedDocs = json.load(open(self.args.out_folder + "stopword_removed_docs.txt", 'r'))[:]

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids, self.args.preprocess)

        # Load list of incomplete queries
        with open(self.args.out_folder + "NP_list.npy", 'rb')as f:
            NP_list = pickle.load(f)

        query_log = np.load(self.args.out_folder+"query_log.npy")
        MRR = 0
        relevant_query_returned = 0
        num_incomplete_queries = 0

        for q_num in range(len(NP_list)):
            for query in NP_list[q_num]:
                # Process query (after you have done spell check)
                processedQuery = self.preprocessQueries([query])[0]

                # Generating autocomplete suggestions
                ordered_queries = self.informationRetriever.autocomplete(processedQuery, self.args.out_folder)
                ordered_queries = ordered_queries[:5]

                popularity = [[query_log[ordered_query[1], 0], ordered_query[1]] for ordered_query in ordered_queries]
                recentness = [[query_log[ordered_query[1], 1], ordered_query[1]] for ordered_query in ordered_queries]

                sim_max = np.max([pair[0] for pair in ordered_queries])
                pop_max = np.max([pair[0] for pair in popularity])
                rec_max = np.max([pair[0] for pair in recentness])

                if sim_max!=0: # If no query contains a word from the user-entered incomplete query
                    num_incomplete_queries += 1
                    query_weighted_rank = {ordered_queries[i][1]:0.5*ordered_queries[i][0]/sim_max for i in range(len(ordered_queries))}
                    for i in range(5):
                        query_weighted_rank[popularity[i][1]] += 0.3*(popularity[i][0]/pop_max)
                        query_weighted_rank[recentness[i][1]] += 0.2*(recentness[i][0]/rec_max)
                    query_weighted_rank = [[q_rank, q_id] for q_id, q_rank in query_weighted_rank.items()]
                    query_weighted_rank.sort(reverse=True)
                    ordered_queries = [pair[1] for pair in query_weighted_rank]

                if q_num+1 in ordered_queries:
                    relevant_query_returned+=1
                    MRR += (1/(1+ordered_queries.index(q_num+1)))

        relevant_query_returned/=num_incomplete_queries
        MRR/=num_incomplete_queries
        return relevant_query_returned, MRR


if __name__=="__main__":
    overall_start = time()

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default = "cranfield/", help = "Path to the dataset folder")
    parser.add_argument('-out_folder', default = "output/", help = "Path to output folder")
    parser.add_argument('-segmenter', default = "punkt", help = "Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default = "ptb", help = "Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action = "store_true", help = "Take custom query as input")
    parser.add_argument("-model", default="vector_space", help = "Choose the type of model for representing doc vectors")
    parser.add_argument("-k", default=200, help="Number of latent dimensions")
    parser.add_argument("-interactive", action = "store_true", help = "Whether the interactive app is used")
    parser.add_argument("-query", default="", help="Custom query input for interactive app")
    parser.add_argument("-preprocess", action = "store_true", help = "If -preprocess is specified, the queries and docs are preprocessed at runtime.")

    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    # Either handle query from user or evaluate on the complete dataset
    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()

    overall_end = time()
    print("Total time taken = {} seconds\n".format(overall_end - overall_start))
