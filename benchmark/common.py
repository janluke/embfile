from pathlib import Path
from typing import NamedTuple, Callable

import numpy

DATA_DIR    = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
TABLES_DIR  = Path(__file__).parent / 'tables'

METHODS = ['median', 'best', 'mean']
DEFAULT_METHOD = 'median'


class SummarizationMethod(NamedTuple):
    summarize: Callable
    desc: str


def get_summarization_method(method, summary):
    if method == 'median':
        summarize = numpy.median
        desc = 'Median of %d tries' % summary['repeat']

    elif method == 'best':
        summarize = min
        desc = 'Best of %d tries' % summary['repeat']

    elif method == 'mean':
        summarize = numpy.mean
        desc = 'Mean of %d tries' % summary['repeat']

    else:
        raise ValueError(method)

    return SummarizationMethod(summarize, desc)


def format_int(n):
    if n % 1_000_000 == 0:
        return '%dM' % (n // 1_000_000)
    if n > 1_000_000 and n % 100_000 == 0:
        return '%.1fM' % (n / 1_000_000)
    if n % 1_000 == 0:
        return '%dK' % (n // 1_000)
    return '{:,}'.format(n)
