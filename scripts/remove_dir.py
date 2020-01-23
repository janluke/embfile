""" Platform-independent equivalent of "rm -rf" to use with make/tox """
import os
import shutil
import sys

path = sys.argv[1]
if os.path.exists(path):
    shutil.rmtree(path)
