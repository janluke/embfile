+---------------------------------+-----------+------------+-----------------------------------+
| Class                           | format_id | Extensions | Description                       |
+=================================+===========+============+===================================+
| :class:`~embfile.TextEmbFile`   | txt       | .txt, .vec |  Glove/fastText format            |
+---------------------------------+-----------+------------+-----------------------------------+
| :class:`~embfile.BinaryEmbFile` | bin       | .bin       |  Google word2vec format           |
+---------------------------------+-----------+------------+-----------------------------------+
| :class:`~embfile.VVMEmbFile`    | vvm       | .vvm       | Custom format storing vocabulary  |
|                                 |           |            | vectors and metadata in separate  |
|                                 |           |            | files inside a TAR                |
+---------------------------------+-----------+------------+-----------------------------------+
