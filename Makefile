PHONY: tests

install-dev:
	poetry install
	ipython kernel install --user --name=samples
