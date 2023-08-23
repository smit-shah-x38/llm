from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

import nltk

nltk.download("punkt")

warnings.filterwarnings(action="ignore")
import gensim
from gensim.models import Word2Vec
import tensorflow as tf

# Load your text data
# sample = open("your_text_file.txt", "r")

import requests

url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
response = requests.get(url)
s = response.text

# s = tf.keras.utils.get_file(
#     "shakespeare.txt",
#     "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
# )

s = s.replace("\n", " ")

# with open(s) as f:
#     lines = f.read().splitlines()
# for line in lines[:20]:
#     print(line)

# s = sample.read()
# f = s.replace("\n", " ")

# Tokenize your text data
data = []
for i in sent_tokenize(s):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

# Train the word2vec model
model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)

# Test the model
print(
    "Cosine similarity between 'alonso' and 'antonio': ",
    model.wv.similarity("alonso", "antonio"),
)

model.save("word2vec.model")
