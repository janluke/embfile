
PACKAGE = embfile

.PHONY: isort
isort:
	isort --recursive src tests setup.py

.PHONY: isort-check
isort-check:
	isort --check-only --diff --recursive src tests setup.py

.PHONY: lint
lint: isort-check
	mypy src tests
	flake8 src tests

.PHONY: build
build:
	pip install --upgrade setuptools wheel twine
	python setup.py clean --all sdist bdist_wheel

.PHONY: check
check:
	twine check dist/*

.PHONY: upload
upload: check
	twine upload --skip-existing dist/*.whl dist/*.gz

.PHONY: test-upload
test-upload: check
	twine upload -r testpypi --skip-existing dist/*.whl dist/*.gz
