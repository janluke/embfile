======
Usage
======

.. contents:: Table of contents
    :local:
    :depth: 2


Opening a file
===============
The core class of the package is the abstract class :class:`~embfile.core.EmbFile`.
Three subclasses are implemented, one per supported format. Each format is associated
with a ``format_id`` (string) and one or multiple file extensions:

.. include:: formats_table.rst

You can open an embedding file either:

- using the constructor of any of the subclasses above::

    from embfile import BinaryEmbFile

    with BinaryEmbFile('GoogleNews-vectors-negative300.bin') as file:
        ...

- or using :meth:`embfile.open`, which by default infers the file format from the file extension::

    import embfile

    with embfile.open('GoogleNews-vectors-negative300.bin') as file:
        print(file)

    """ Will print:
    BinaryEmbFile (
      path = GoogleNews-vectors-negative300.bin,
      vocab_size = 3000000,
      vector_size = 300
    )
    """

  You can force a particular format passing the ``format_id`` argument.

All the ``path`` arguments can either be of type string or :class:`pathlib.Path`.
Object attributes storing paths are always :class:`pathlib.Path`, not strings.


Shared arguments
----------------
All the :class:`~embfile.EmbFile` subclasses support two `optional` arguments (that you can safely
pass to ``embfile.open`` as well):

- ``out_dtype`` (numpy.dtype) -- if provided, all vectors read from the file
  are converted to this data type (if needed) before being returned;

- ``verbose`` (bool) --  sets the `default value` of the ``verbose`` argument exposed by all
  time-consuming ``EmbFile`` methods; when ``verbose is True``, progress bars are displayed
  by default; you can always pass ``verbose=False`` to a method to disable console output.


Format-specific arguments
-------------------------

For format-specific arguments, check out the specific class documentation:

.. autosummary::
    ~embfile.BinaryEmbFile
    ~embfile.TextEmbFile
    ~embfile.VVMEmbFile

You can pass format-specific arguments to ``embfile.open`` too.


Compressed files
----------------
How to handle compression is left to ``EmbFile`` subclasses. As a general rule,
a concrete ``EmbFile`` requires non-compressed files unless the opposite is
specified in its docstring. Anyway, in most cases, you want to work on non-compressed
files because it's much faster (of course).

``embfile`` provide utilities to work with compression in the submodule
:mod:`~embfile.compression`; the following functions can be used
(or imported) directly from the root module:

.. autosummary::
    embfile.extract
    embfile.extract_if_missing

Lazy (on-the-fly) decompression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, ``TextEmbFile`` is the only format that allows you to open a compressed file
directly and to decompress it "lazily" while reading it. Lazy decompression works for
all compression formats but zip. For uniformity of behavior, you can still open zipped
files directly but, under the hood, the file will be fully extracted to a temporary file
before starting reading it.

Lazy decompression makes sense only if you only want to perform a single pass through the
file (e.g. you are converting the file); indeed, every new operation (that requires to
create a new :ref:`file reader<readers>`) requires to (lazily) decompress the file again.


Registering new formats or file extensions
------------------------------------------
.. currentmodule:: embfile

Format ID and file extensions of each registered file format are stored in the global object
``embfile.FORMATS``. To associate a file extension to a registered format you can
use :func:`associate_extension`:

.. runblock:: pycon

    >>> import embfile
    >>> embfile.associate_extension(ext='.w2v', format_id='bin')
    >>> print(embfile.FORMATS)

To register a new format (see `Implementing a new format`_), you can use the class decorator
:func:`register_format`::

    @embfile.register_format(format_id='hdf5', extensions=['.h5', '.hdf5'])
    class HDF5EmbFile(EmbFile):
        # ...


Loading word vectors
====================
Loading specific word-vectors
-----------------------------
.. autosummary::
    ~EmbFile.load
    ~EmbFile.find
    ~EmbFile.loader

::

    word2vec = f.load(['hello', 'world'])  # raises KeyError if any word is missing

    word2vec, missing_words = f.find(['hello', 'world', 'missingWord'])


You should prefer ``loader`` to ``find`` when you want to store the vectors
*directly* into some custom data structure without wasting time and memory for building an
intermediate dictionary. For example, :func:`~build_matrix` uses ``loader`` to
load the vectors directly into a numpy array.

Here's how you use a loader::

    data_structure = MyCustomStructure()
    for word, vector in file.loader(many_words):
        data_structure[word] = vector

If you're interested in ``missing_words``::

    data_structure = MyCustomStructure()
    loader = file.loader(many_words)
    for word, vector in loader:
        data_structure[word] = vector
    print('Missing words:', loader.missing_words)


Loading the entire file in memory
---------------------------------
.. currentmodule:: embfile.core
.. autosummary::
    ~EmbFile.to_dict
    ~EmbFile.to_list


Building a matrix
=================
The docstring of :meth:`embfile.build_matrix` contains everything you need to know
to use it. Here, we'll give some examples through an IPython session.

First, we'll generate a dummy file with only three vectors:

.. ipython:: python

    import tempfile
    from pathlib import Path
    import numpy as np
    import embfile
    from embfile import VVMEmbFile

    word_vectors = [
        ('hello', np.array([1, 2, 3])),
        ('world', np.array([4, 5, 6])),
        ('!',     np.array([7, 8, 9]))
    ]

    path = Path(tempfile.gettempdir(), 'file.vvm')
    VVMEmbFile.create(path, word_vectors, overwrite=True, verbose=False)

Let's build a matrix out of a list of words. We'll use the default oov_initializer
for initializing the vectors for out-of-file-vocabulary words:

.. ipython:: python

    words = ['hello', 'ciao', 'world', 'mondo']
    with embfile.open(path, verbose=False) as f:
        result = embfile.build_matrix(
            f, words,
            start_index=1,   # map the first word to the row 1 (default is 0)
        )

    # result belongs to a class that extends NamedTuple
    print(result.pretty())
    result.matrix
    result.word2index
    result.missing_words

Now, we'll build a matrix from a dictionary ``{word: index}``. We'll use a custom
``oov_initializer`` this time.

.. ipython:: python

    word2index = {
        'hello': 1,
        'ciao': 3,
        'world': 4,
        'mondo': 5
    }

    with embfile.open(path, verbose=False) as f:
        def custom_initializer(shape):
            scale = 1 / np.sqrt(f.vector_size)
            return np.random.normal(loc=0, scale=scale, size=shape)
        result = embfile.build_matrix(f, word2index, oov_initializer=custom_initializer)

    print(result.pretty())
    result.matrix
    result.word2index
    result.missing_words

See :mod:`embfile.initializers` for checking out the available initializers.


Iteration
=========
.. currentmodule:: embfile.core

.. _readers:

File readers
--------------------------
Efficient iteration of the file is implemented by format-specific readers.

.. autosummary::
    EmbFileReader

A new reader for a file can be created using the method :meth:`~EmbFile.reader`. Every
method that requires to iterate the file entries sequentially uses this method to
create a new reader.

You usually won't need to use a reader directly because ``EmbFile`` defines quicker-to-use
methods that use a reader for you. If you are interested, the docstring is pretty detailed.

Dict-like methods
-----------------
The following methods are wrappers of :meth:`~EmbFile.reader`.
Keep in mind that every time you use these methods, you are creating a new file reader
and items are read from disk (the vocabulary may be loaded in memory though,
as in VVM files).

.. autosummary::
    ~EmbFile.words
    ~EmbFile.vectors
    ~EmbFile.word_vectors
    ~EmbFile.filter

**Don't** use ``word_vectors()`` if you want to filter the vectors based on a condition
on words: it'll read vectors for all words you read, even those that don't meet the condition.
Use ``filter`` instead.


Creating/converting a file
==========================
.. currentmodule:: embfile.core

Each subclass of ``EmbFile`` implements the following two class methods:

.. autosummary::
    ~EmbFile.create
    ~EmbFile.create_from_file

Examples of file creation
---------------------------
You can create a new file either from:

- a dictionary ``{word: vector}``
- an iterable of ``(word, vector)`` tuples; the iterable can also be an iterator/generator.

For example::

    import numpy as np
    from embfile import VVMEmbFile

    word_vectors = {
        "hello": np.array([0.1, 0.2, 0.3]),
        "world": np.array([0.4, 0.5, 0.6])
        # ... a lot more word vectors
    }

    VVMEmbFile.create(
        '/tmp/dummy.vvm.gz',
        word_vectors,
        dtype='<2f',      # store numbers as little-endian 2-byte float
        compression='gz'  # compress with gzip
    )


Example of file conversions
------------------------------
Let's convert a textual file to a vvm file. The following will generate a compressed
vvm file in the same folder of the textual file (and with a proper file extension)::

    from embfile import VVMEmbFile

    with embfile.open('path/to/source/file.txt') as src_file:
        dest_path = VVMEmbFile.create_from_file(src_file, compression='gz')

    # dest_path  == Path('path/to/source/file.vvm.gz')


Implementing a new format
=========================
If you ever feel the need for implementing a new format, it's fairly easy to integrate
your custom format in this library and to test it. My suggestion is:

#. grab the template below
#. read :class:`~embfile.core.EmbFile` docstring
#. look at existing implementations in the ``embfile.formats`` subpackage
#. for testing, see how they are tested in ``tests/test_files.py``

You are highly suggested to use a IDE of course.

.. literalinclude:: extending.py

This'll print::

    """
    Class          Format ID    Extensions
    -------------  -----------  ------------
    BinaryEmbFile  bin          .bin
    TextEmbFile    txt          .txt, .vec
    VVMEmbFile     vvm          .vvm
    CustomEmbFile  custom       .cst, .cust
    """
