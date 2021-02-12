# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'embfile'
author = 'Gianluca Gippetto'
year = '2019'
copyright = '{0}, {1}'.format(year, author)

# The short X.Y version
version = '0.1.0'
# The full version, including alpha/beta/rc tags
release = version

extlinks = {
    'issue': ('https://github.com/janLuke/embfile/issues/%s', '#'),
    'pr': ('https://github.com/janLuke/embfile/pull/%s', 'PR #'),
}

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib_trio',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx_autorun',
    'sphinx_autodoc_typehints',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'trac'    # styles preview: https://help.farbox.com/pygments.html


# -- Options for HTML output -------------------------------------------------

html_theme = 'guzzle_sphinx_theme'  # Choices: sphinx_rtd_theme, guzzle_sphinx_theme
html_use_smartypants = True         # smartypants replaces ascii chars with better unicode chars

if html_theme == 'guzzle_sphinx_theme':
    import guzzle_sphinx_theme
    extensions += ["guzzle_sphinx_theme"]
    html_theme_path = guzzle_sphinx_theme.html_theme_path()

    # Add any paths that contain custom static files (such as style sheets) here,
    # relative to this directory. They are copied after the builtin static files,
    # so a file named "default.css" will overwrite the builtin "default.css".
    html_static_path = ['_static']
    html_style = 'css/custom_guzzle.css'

    html_theme_options = {
        # Set the path to a special layout to include for the homepage
        # "index_template": "special_index.html",

        # Set the name of the project to appear in the left sidebar.
        "project_nav_name": project,

        # Visible levels of the global TOC; -1 means unlimited
        # "globaltoc_depth": 3,

        # If False, expand all TOC entries
        # "globaltoc_collapse": False,

        # If True, show hidden TOC entries
        # "globaltoc_includehidden": False,
    }

    html_sidebars = {
        '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']
    }

elif html_theme == 'sphinx_rtd_theme':
    html_sidebars = {
       '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
    }

else:
    raise ValueError('invalid html_theme')


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'embfiledoc'


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'embfile', 'embfile Documentation',
     [author], 1)
]


# -- Extension configuration -------------------------------------------------

# ==========================
#  Inter-sphinx
# ==========================
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
}

# ==========================
#  Napoleon
# ==========================
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = True

# ==========================
#  Auto-doc
# ==========================
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': None,
    'undoc-members': None,
    'show-inheritance': None,
    'inherited-members': False,
    'special-members': '__contains__, __getitem__'
}


def run_apidoc(app):
    """Generage API documentation"""
    import better_apidoc
    better_apidoc.APP = app
    file_dir = os.path.abspath(os.path.dirname(__file__))
    better_apidoc.main([
        'better-apidoc',
        '-t', os.path.join(file_dir, '_templates', 'apidoc'),
        '--force',
        '--no-toc',
        '--separate',
        '-o', os.path.join(file_dir, 'api'),
        os.path.abspath(os.path.join(file_dir, '..', 'src', 'embfile')),
    ])
    # replace the title of the main api file from $(PACKAGE) to $(API_SECTION)
    api_srt_path = os.path.join(file_dir, 'api', 'embfile.rst')
    with open(api_srt_path) as f:
        lines = f.readlines()
    lines[0] = 'embfile API\n'
    lines[1] = '=' * len(lines[0]) + '\n'
    with open(api_srt_path, 'w') as fout:
        fout.writelines(lines)


def setup(app):
    app.connect('builder-inited', run_apidoc)

# Suppress "WARNING: more than one target found for cross-reference "
# https://github.com/sphinx-doc/sphinx/issues/3866#issuecomment-311181219
# class PatchedPythonDomain(PythonDomain):
#     def resolve_xref(cls, env, fromdocname, builder, typ, target, node, contnode):
#         if 'refspecific' in node:
#             del node['refspecific']
#         return super(PatchedPythonDomain, cls).resolve_xref(
#             env, fromdocname, builder, typ, target, node, contnode)
#
#
# def setup(app):
#     app.override_domain(PatchedPythonDomain)
