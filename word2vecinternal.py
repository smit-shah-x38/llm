import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1
            )
            (
                negative_sampling_candidates,
                _,
                _,
            ) = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling",
            )

            # Build context and label vectors (for one target word)
            context = tf.concat(
                [tf.squeeze(context_class, 1), negative_sampling_candidates], 0
            )
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

with open(path_to_file) as f:
    lines = f.read().splitlines()
for line in lines[:20]:
    print(line)

text_ds = tf.data.TextLineDataset(path_to_file).filter(
    lambda x: tf.cast(tf.strings.length(x), bool)
)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(string.punctuation), ""
    )


# Define the vocabulary size and the number of words in a sequence.
vocab_size = 4096
sequence_length = 10

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()


sequences = list(text_vector_ds)
print(len(sequences))

for seq in sequences[:5]:
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

targets, contexts, labels = generate_training_data(
    sequences=sequences, window_size=2, num_ns=4, vocab_size=vocab_size, seed=SEED
)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print("\n")
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")


BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding"
        )
        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=num_ns + 1
        )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        # dots: (batch, context)
        return dots


embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

weights = word2vec.get_layer("w2v_embedding").get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open("vectors.tsv", "w", encoding="utf-8")
out_m = io.open("metadata.tsv", "w", encoding="utf-8")

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write("\t".join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()

try:
    from google.colab import files

    files.download("vectors.tsv")
    files.download("metadata.tsv")
except Exception:
    pass
