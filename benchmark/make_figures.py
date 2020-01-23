import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from common import (
    DEFAULT_METHOD,
    FIGURES_DIR,
    METHODS,
    format_int,
    get_summarization_method
)

plt.style.use('seaborn')

mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.titlepad'] = 12
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['axes.labelsize'] = 11

mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.facecolor'] = 'ffffff'
mpl.rcParams['legend.framealpha'] = .8
mpl.rcParams['legend.borderpad'] = .6


def plot_results(summary, method=DEFAULT_METHOD):
    data = summary['data']
    vocab_size = summary['vocab_size']
    vector_size = summary['vector_size']

    summarize_times, description = get_summarization_method(method, summary)

    classes = sorted(set(cls for cls, _ in data), key=lambda cls: cls.__name__)
    query_sizes = sorted(set(qsize for _, qsize in data))

    for cls in classes:
        stat = [summarize_times(data[cls, qsize]) for qsize in query_sizes]
        plt.plot(query_sizes, stat, 'o-')

    plt.legend([cls.__name__ for cls in classes])

    description += ' [files with {} vectors of length {}]'.format(format_int(vocab_size),
                                                            format_int(vector_size))
    plt.title(description)
    plt.xlabel('Number of words to load')
    plt.ylabel('Time (sec)')
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.xticks(query_sizes, labels=[format_int(s) for s in query_sizes])


def save_plot(basename, out_dir=FIGURES_DIR, formats=['.svg']):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = (out_dir / basename).with_suffix(fmt)
        plt.savefig(str(path))  # str() solves matplotlib bug
        paths.append(path)
    return paths


def remake_all_figures(method):
    import pickle
    from glob import glob

    result_dir = Path(__file__).parent / 'results'
    for file_path in glob(str(result_dir / '*.pkl')):
        print('Processing file', file_path)
        with open(file_path, 'rb') as f:
            summary = pickle.load(f)

        plt.figure()
        plot_results(summary, method=method)
        basename = Path(file_path).stem + '_' + method
        plot_paths = save_plot(basename)

        for plot_path in plot_paths:
            print('   ', plot_path)
        print('')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=DEFAULT_METHOD, choices=METHODS)
    args = parser.parse_args()

    remake_all_figures(method=args.method)
