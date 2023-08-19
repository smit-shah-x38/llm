# Import libraries
import gensim
import torch

# Define some sentences
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word", "embeddings", "are", "useful", "for", "context", "recognition"],
    ["Python", "is", "a", "great", "language", "for", "AI"],
]

# Create a Word2Vec model
model = gensim.models.Word2Vec(sentences, min_count=1)


# Define a function to make context vector
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


# Define a word to index mapping
word_to_ix = {word: i for i, word in enumerate(model.wv.index_to_key)}

# Define a context and a target word
context = ["I", "love"]
target = "natural"

# Make a context vector
context_vector = make_context_vector(context, word_to_ix)

# Print the context vector
print(context_vector)

# Get the word embedding for the context vector
word_embedding = model.wv[context_vector]

# Print the word embedding
print(word_embedding)

# Get the probability of the target word
prob = model.predict_output_word(context + [target])

# Print the probability
print(prob)
