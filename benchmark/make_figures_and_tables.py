from common import METHODS, DEFAULT_METHOD
from make_figures import remake_all_figures
from make_table import remake_all_tables


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=DEFAULT_METHOD, choices=METHODS)
    args = parser.parse_args()

    remake_all_figures(method=args.method)
    remake_all_tables(method=args.method)
