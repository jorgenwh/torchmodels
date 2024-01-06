.PHONY: all install dev-install uninstall clean

all: install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall torchmodels 

clean:
	$(RM) -rf build torchmodels.egg-info *.so dist/ torchmodels/__pycache__/
