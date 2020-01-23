python benchmark.py -g 1000000 100 -q 1000 50000 150000 300000 -r 5 --no-plot
python benchmark.py -g 1000000 300 -q 1000 50000 150000 300000 -r 5 --no-plot
python benchmark.py -g 3000000 100 -q 1000 50000 150000 300000 -r 5 --no-plot
python benchmark.py -g 3000000 300 -q 1000 50000 150000 300000 -r 5 --no-plot
python make_figures_and_tables.py --method best
python make_figures_and_tables.py --method median
