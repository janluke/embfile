========
Overview
========

.. start-badges

|tests-status| |codecov| |version| |supported-versions| |docs|

.. |docs| image:: https://readthedocs.org/projects/embfile/badge/?style=flat
    :target: https://readthedocs.org/projects/embfile
    :alt: Documentation Status

.. |tests-status| image:: https://github.com/janLuke/embfile/workflows/Tests/badge.svg
    :alt: Tests status
    :target: https://github.com/janLuke/embfile/actions?query=workflow%3ATests

.. |codecov| image:: https://codecov.io/github/janLuke/embfile/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/janLuke/embfile

.. |version| image:: https://img.shields.io/pypi/v/embfile.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/embfile

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/embfile.svg
    :alt: Supported versions
    :target: https://pypi.org/project/embfile

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/embfile.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/embfile

.. end-badges

A package for working with files containing word embeddings (aka word vectors).
Written for:

#. providing a common interface for different file formats;
#. providing a flexible function for building "embedding matrices" that you can use
   for initializing the `Embedding` layer of your deep learning model;
#. taking as less RAM as possible: no need to load 3M vectors like with
   `gensim.load_word2vec_format` when you only need 20K;
#. satisfying my (inexplicable) urge of writing a Python package.


Features
========
- Supports textual and Google's binary format plus a custom convenient format (.vvm)
  supporting constant-time access of word vectors (by word).

- Allows to easily implement, test and integrate new file formats.

- Supports virtually any text encoding and vector data type (though you should
  probably use only UTF-8 as encoding).

- Well-documented and type-annotated (meaning great IDE support).

- Extensively tested.

- Progress bars (by default) for every time-consuming operation.


Installation
============
::

    pip install embfile


Quick start
===========

.. code-block:: python

    import embfile

    with embfile.open("path/to/file.bin") as f:     # infer file format from file extension

        print(f.vocab_size, f.vector_size)

        # Load some word vectors in a dictionary (raise KeyError if any word is missing)
        word2vec = f.load(['ciao', 'hello'])

        # Like f.load() but allows missing words (and returns them in a Set)
        word2vec, missing_words = f.find(['ciao', 'hello', 'someMissingWord'])

        # Build a matrix for initializing an Embedding layer either from
        # a list of words or from a dictionary {word: index}. Handles the
        # initialization of eventual missing word vectors (see "oov_initializer")
        matrix, word2index, missing_words = embfile.build_matrix(f, words)

Examples
========
The examples shows how to use embfile to initialize the ``Embedding`` layer of
a deep learning model. They are just illustrative, don't skip the documentation.

- `Keras using Tokenizer <https://github.com/janLuke/embfile/blob/master/examples/keras_with_Tokenizer.py>`_
- `Keras using TextVectorization <https://github.com/janLuke/embfile/blob/master/examples/keras_with_TextVectorization.py>`_
  (tensorflow >= 2.1)

.. if-doc-stop-here

Documentation
=============
Read the full documentation at https://embfile.readthedocs.io/.
