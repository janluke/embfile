import os
import pickle
from glob import glob
from pathlib import Path

from common import (
    get_summarization_method,
    format_int,
    METHODS,
    DEFAULT_METHOD,
    TABLES_DIR
)

CSV_DELIM = ','

TABLE_FMT = """
.. csv-table:: {title}
    :delim: %s
    :header: {input_sizes}
    :align: center
    :widths: auto

{data}
""" % CSV_DELIM


def make_result_table(summary, method=DEFAULT_METHOD):
    data = summary['data']
    classes = sorted(set(cls for cls, _ in data), key=lambda cls: cls.__name__)
    query_sizes = sorted(set(qsize for _, qsize in data))
    summarize_times, description = get_summarization_method(method, summary)

    table_rows = []
    for cls in classes:
        row = ['``%s``' % cls.__name__]
        times_for_increasing_size = (data[cls, qsize] for qsize in query_sizes)
        reported_times = [summarize_times(times) for times in times_for_increasing_size]
        row += ('%.1f' % t for t in reported_times)
        table_rows.append(row)

    input_sizes = CSV_DELIM.join([''] + [format_int(size) for size in query_sizes])
    csv_lines = (CSV_DELIM.join(row) for row in table_rows)
    indentation = ' ' * 4
    indented_lines = (indentation + line for line in csv_lines)
    data_str = '\n'.join(indented_lines)

    return TABLE_FMT.format(title='', input_sizes=input_sizes, data=data_str)


def save_result_table(data, basename, method=DEFAULT_METHOD, outdir=TABLES_DIR):
    table = make_result_table(data, method=method)
    table_path = outdir / (basename + '.rst')
    os.makedirs(outdir, exist_ok=True)
    with open(table_path, 'wt') as f:
        f.write(table)
    return table


def remake_all_tables(method):
    result_dir = Path(__file__).parent / 'results'
    for file_path in glob(str(result_dir / '*.pkl')):

        with open(file_path, 'rb') as f:
            exp = pickle.load(f)

        basename = Path(file_path).stem + '_' + method
        table = save_result_table(exp, method=method, basename=basename)
        print(table)
        print('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=DEFAULT_METHOD, choices=METHODS)
    args = parser.parse_args()

    remake_all_tables(method=args.method)
