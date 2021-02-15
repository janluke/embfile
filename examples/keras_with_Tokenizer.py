from pprint import pprint

import numpy

import embfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# This is just for testing the script without a real file on disk
EMBEDDING_FILE_PATH = 'dummy_embeddings.txt'
with open(EMBEDDING_FILE_PATH, 'w', encoding='utf-8') as f:
    f.write('5 5\n'
            'this 1 1 1 1 1\n'
            'is 2 2 2 2 2\n'
            'time 3 3 3 3 3\n'
            'document 4 4 4 4 4\n'
            'there 5 5 5 5 5')


# Some dummy training documents
training_docs = [
    'This is a document',
    'This is another document',
    'There was a time... another document',
]

# Fit a Tokenizer on the training data and print the {word -> index} dictionary
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_docs)
print('Word index:')
pprint(tokenizer.word_index, sort_dicts=False)
print('')

# Convert the documents to sequences of indexes
MAX_SEQ_LENGTH = 7
sequences = pad_sequences(
    tokenizer.texts_to_sequences(training_docs),
    maxlen=MAX_SEQ_LENGTH,
)
print('Sequences:', sequences, sep='\n', end='\n\n')

# Use embfile to create an embedding matrix
# See the docs for more info about embfile.build_matrix()
with embfile.open(EMBEDDING_FILE_PATH, encoding='utf-8', verbose=False) as emb_file:
    # I'm not unpacking the result here because it has a method for
    # pretty-printing itself, useful for a tutorial like this
    result = embfile.build_matrix(emb_file, tokenizer.word_index)
    print('Embedding matrix:', result.pretty(), sep='\n', end='\n\n')

    emb_matrix, word2index, missing_words = result
    assert word2index == tokenizer.word_index
    # word2index is redundant when only when you pass a dict to build_matrix(),
    # but it's useful when you instead pass a list of words.

# Initialize the Embedding layer with pre-trained word vectors.
# To configure the Embedding layer, check out the Keras documentation.
num_tokens = len(tokenizer.word_index) + 1
embedding_dim = emb_matrix.shape[1]
embedding_layer = Embedding(
    input_dim=num_tokens,
    output_dim=embedding_dim,
    input_length=MAX_SEQ_LENGTH,
    embeddings_initializer=keras.initializers.Constant(emb_matrix),
    trainable=False,  # if you don't want to fine-tune the embeddings
)

# Just for testing...
inputs = keras.Input(shape=(MAX_SEQ_LENGTH,))
outputs = embedding_layer(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
print('\nEncode the first sequence:', sequences[0])
with numpy.printoptions(precision=3):
    print(model(sequences[0]))
