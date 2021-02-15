import numpy

import embfile
from tensorflow import keras
from tensorflow.python.keras.layers import (
    Embedding,
    TextVectorizationV2 as TextVectorization,
)


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
training_docs = numpy.array([
    'This is a document',
    'This is another document',
    'There was a time... another document',
])

# Fit a Tokenizer on the training data and print the {word -> index} dictionary
MAX_SEQ_LENGTH = 7
text_vectorizer = TextVectorization(
    output_sequence_length=MAX_SEQ_LENGTH,
)
text_vectorizer.adapt(training_docs)
sequences = text_vectorizer(training_docs)
print('Sequences:', sequences, sep='\n', end='\n\n')

# Get the vocabulary. Note that the first 2 "words" are '' and [UNK]
vocab = text_vectorizer.get_vocabulary()
print('Vocabulary:', vocab, end='\n\n')

# Build the embedding matrix
with embfile.open(EMBEDDING_FILE_PATH, encoding='utf-8', verbose=False) as emb_file:
    # I'm not unpacking the result here because it has a method for
    # pretty-printing itself, useful for a tutorial like this
    result = embfile.build_matrix(
        emb_file,
        words=vocab[1:],   # don't consider '' a word
        start_index=1,     # leave matrix[0], i.e. the padding vector, to zeros
    )
    print('Embedding matrix:', result.pretty(), sep='\n', end='\n\n')
    emb_matrix, word2index, missing_words = result

# Initialize the Embedding layer with pre-trained word vectors.
# To configure the Embedding layer, check out the Keras documentation.
embedding_dim = emb_matrix.shape[1]
embedding_layer = Embedding(
    input_dim=len(vocab),
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
