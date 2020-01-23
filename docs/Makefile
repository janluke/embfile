
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# My vars:
PACKAGE = embfile
PACKAGE_ROOT = ../src/$(PACKAGE)
APIDOC_TEMPLATES = $(SOURCEDIR)/_templates/apidoc
APIDIR =  $(SOURCEDIR)/api
API_SECTION = embfile API

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#=================
# Added command
#=================
#.PHONY: apidoc
#apidoc:
#	better-apidoc -o $(APIDIR) $(PACKAGE_ROOT) -t $(APIDOC_TEMPLATES) -f --separate --no-toc
#	# replace the title of the main api file from $(PACKAGE) to $(API_SECTION)
#	sed -i "1s/.*/$(API_SECTION)/" $(APIDIR)/$(PACKAGE).rst
#	sed -i "2s/.*/===================/" $(APIDIR)/embfile.rst

.PHONY: clean-api
clean-api:
	python ../scripts/remove_dir.py $(APIDIR)

.PHONY: clean
clean:
	python ../scripts/remove_dir.py $(BUILDDIR)

.PHONY: html
html:
	sphinx-build -b html $(SOURCEDIR) $(BUILDDIR)

.PHONY: docs
docs: clean clean-api html

.PHONY: css
css:
	cp _static/css/custom_guzzle.css $(BUILDDIR)/_static/css/custom_guzzle.css
