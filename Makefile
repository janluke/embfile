
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

.PHONY: sdist
build:
	python setup.py clean --all sdist bdist_wheel

.PHONY: check
check:
	twine check dist/*

.PHONY: upload
upload:
	twine register dist/*
	twine upload --skip-existing dist/*.whl dist/*.gz dist/*.zip

.PHONY: upload-test
upload-test:
	twine register dist/*
	twine upload --skip-existing dist/*.whl dist/*.gz dist/*.zip \
	             --repository-url https://test.pypi.org/legacy/
