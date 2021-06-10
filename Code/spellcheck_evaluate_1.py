import pandas as pd
import json
import string
import random
import re
import numpy as np
import pandas as pd
from bigramSpellCheck import BigramSpellCheck

# Load the documents
DATASET = "Cranfield_Dataset/cranfield/"
docs_json = json.load(open(DATASET + "cran_docs.json", 'r'))[:]
doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]

# Form the vocabulary
all_words = []
for doc in docs:
    all_words = all_words + re.sub("[^a-z ]+", " ", doc).split()

# A list of all the unique words in the corpus
vocabulary = list(set(all_words))

letters = string.ascii_lowercase
generated_words = []
correct_words = []

print("Reached1")

#### GENERATING EVALUATION DATASET
for word in vocabulary:
    length = len(word)
    if length < 2:
        continue
    # Random insertion
    index = random.randint(0, length - 1)
    generated_words.append(word[:index] + random.choice(string.ascii_lowercase) + word[index:])
    # Random substitution
    generated_words.append(word[:index] + random.choice(string.ascii_lowercase) + word[index+1:])
    # Random deletion
    index = random.randint(0, length - 2)
    generated_words.append(word[:index]+word[index+1:])
    # Transposition
    generated_words.append(word[:index] + word[index+1] + word[index] + word[index+2:])

    correct_words = correct_words + [word]*4

df = pd.DataFrame(data={"correct_word": correct_words, "generated_words": generated_words})
df.to_csv("Output/spellcheck_eval_list.csv", index= False)

print("Reached2")

#### EVALUATION
df = pd.read_csv("Output/spellcheck_eval_list.csv")

generated_words = list(df["generated_words"])
correct_words = list(df["correct_word"])
obj = BigramSpellCheck(docs)

model_corrections = []
for word in generated_words:
    model_corrections.append(obj.correct_word(word))

acc = np.mean(np.array(model_corrections)==np.array(correct_words))

incor_preds = df.iloc[np.where(np.array(model_corrections)!=np.array(correct_words))]
incor_preds['model_pred'] = [model_corrections[i] for i in np.where(np.array(model_corrections)!=np.array(correct_words))[0].tolist()]

incor_preds.to_csv("Output/spellcheck_incor_preds.csv")

print("\nAccuracy of spelling correction = {} %\n".format(acc * 100.0))
