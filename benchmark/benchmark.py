# coding: utf-8

import argparse
import gc
import os
import pickle
import random
import sys
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy
from common import (DATA_DIR, DEFAULT_METHOD, FIGURES_DIR, METHODS, RESULTS_DIR)
from make_figures import (
    plot_results
)

import embfile
from embfile import (BinaryEmbFile, TextEmbFile, VVMEmbFile)

parser = argparse.ArgumentParser()
input_file = parser.add_mutually_exclusive_group(required=True)

input_file.add_argument(
    '--file-path', '-i', help='path to the file with no extension')

input_file.add_argument(
    '--generate', '-g', nargs=2, type=int, default=[None, None],
    help='vocab_size and vector_size (space-separated)')

parser.add_argument(
    '--fmts', '-f', nargs='+', default=embfile.FORMATS.format_ids(),
    help='Formats to test (IDs)')

parser.add_argument(
    '--query-sizes', '-q', nargs='+', type=int,
    default=[1_000, 250_000, 500_000])

parser.add_argument(
    '--missing-words', type=int, default=1)

parser.add_argument(
    '--repeat', '-r', type=int, default=5)

parser.add_argument(
    '--method', '-m', choices=METHODS, default=DEFAULT_METHOD)

parser.add_argument('--no-plot', action='store_true')

args = parser.parse_args()

classes_to_test = [embfile.FORMATS.id_to_class[fid] for fid in args.fmts]
vocab_size, vector_size = args.generate
num_missing_words = args.missing_words
query_sizes = sorted(filter(lambda s: s <= vocab_size, args.query_sizes))
if not query_sizes:
    sys.exit('Error: no query size smaller than vocab_size in --query-sizes')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# Generate dummy data
def generate_vocab(vocab_size):
    return ['word_%d_abc' % i for i in range(vocab_size)]


def generate_pairs(vocab, vector_size, dtype='float32', seed=12345):
    # generator that generate always exactly the same word_vectors
    numpy.random.seed(seed)
    return ((word, numpy.random.rand(vector_size).astype('float32'))
            for word in vocab)


def dummy_file_path(cls, vocab_size=vocab_size, vector_size=vector_size):
    basename = DATA_DIR / '{}_{}_{}'.format(cls.__name__, vocab_size, vector_size)
    return basename.with_suffix(cls.DEFAULT_EXTENSION)


def real_file_path(cls):
    return Path(args.file_path).with_suffix(cls.DEFAULT_EXTENSION)


if args.file_path:  # use real files
    benchmark_name = Path(args.file_path).name
    file_path = real_file_path
    for cls in classes_to_test:
        if not file_path(cls).exists():
            sys.exit('The file %s does not exist' % file_path(cls))

    for cls in [VVMEmbFile, BinaryEmbFile, TextEmbFile]:  # VVM is the fastest for reading vocab
        if file_path(cls).exists():
            with cls(file_path(cls)) as f:
                vocab = list(f.words())
            vocab_size = len(vocab)
            vector_size = f.vector_size

else:  # use generated files
    benchmark_name = '%d_%d__%d_reps' % (vocab_size, vector_size, args.repeat)
    file_path = dummy_file_path
    print('Generating the vocabulary')
    vocab = generate_vocab(vocab_size)

    for cls in classes_to_test:
        cls_path = dummy_file_path(cls)
        if not cls_path.exists():
            print('File for class %s is missing.' % cls.__name__)
            print('Creating %s file' % cls.DEFAULT_EXTENSION)
            pairs = generate_pairs(vocab, vector_size)  # generate always the same vectors
            cls.create(cls_path, pairs, vocab_size=vocab_size)

print('Generating queries...')
full_query = random.sample(vocab, k=query_sizes[-1])
missing_words = ['<<missing_%d>>' % i for i in range(num_missing_words)]

# Table
COLUMNS = ['Class', 'Query Size', 'Best', 'Median', 'MedianAbsDev']
class_col_width = 2 + max(len(cls.__name__) for cls in classes_to_test)
WIDTHS = (class_col_width,) + tuple(4 + len(c) for c in COLUMNS[1:])
HEADER_FMT = '{:<%ds} {:^%ds} {:^%ds} {:^%ds} {:^%ds}' % WIDTHS
ROW_FMT = '{:<%ds} {:^%dd} {:^%d.2f} {:^%d.2f} {:^%d.2f}' % WIDTHS
HARD_LINE = ' '.join('=' * w for w in WIDTHS)
SOFT_LINE = ' '.join('-' * w for w in WIDTHS)


def print_header():
    print(HARD_LINE)
    print(HEADER_FMT.format(*COLUMNS))
    print(HARD_LINE)


def print_row(row):
    assert len(row) == len(COLUMNS)
    print(ROW_FMT.format(*row))


print_header()

result = {(cls, size): None for size in query_sizes for cls in classes_to_test}

for query_size in query_sizes:
    if query_size > vocab_size:
        print('Skipping query size %d (> vocab size)' % query_size)
        continue
    query = full_query[:query_size]
    query += missing_words

    for cls in classes_to_test:

        path = file_path(cls)
        times = []
        for i in range(args.repeat):
            gc.collect()
            start = default_timer()
            with cls(path, verbose=0) as file:
                file.find(query)
            elapsed = default_timer() - start
            times.append(elapsed)

        result[cls, query_size] = times
        median = numpy.median(times)
        median_abs_dev = numpy.median([abs(t - median) for t in times])
        print_row([cls.__name__, query_size,
                   min(times), median, median_abs_dev])

    if query_size == query_sizes[-1]:
        print(HARD_LINE)
    else:
        print(SOFT_LINE)

# Save results
pkl_path = (RESULTS_DIR / benchmark_name).with_suffix('.pkl')
summary = dict(data=result, vocab_size=vocab_size, vector_size=vector_size, repeat=args.repeat)
with open(pkl_path, 'wb') as f:
    pickle.dump(summary, f)

if not args.no_plot:
    plot_results(summary)
    plt.show()
