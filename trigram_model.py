from nltk.tokenize import RegexpTokenizer
import numpy as np

f = open("dataset.txt", "r", encoding="utf8")

tokenizer = RegexpTokenizer("[\w']+")
tokens = tokenizer.tokenize(f.read())

trigram_model = {}
n = 2

for i in range(len(tokens) - n):
    key = ' '.join(tokens[i:i + n])
    if key not in trigram_model.keys():
        trigram_model[key] = []
    trigram_model[key].append(tokens[i + n])

model = {}
unsorted_model = {}

for word1_word2 in trigram_model:
    unique_classes, counts_unique_classes = np.unique(trigram_model[word1_word2], return_counts=True)
    for i in range(len(unique_classes)):
        x = float(counts_unique_classes[i]) / (float(len(trigram_model[word1_word2])) * 1.0)
        unsorted_model[word1_word2, unique_classes[i]] = x
        if word1_word2 not in model:
            model[word1_word2] = []
        model[word1_word2].append([x, unique_classes[i]])

    model[word1_word2] = sorted(model[word1_word2], reverse=True)
