.PHONY: all install uninstall clean

all: install

install: clean
	pip install -e .

uninstall: clean
	pip uninstall torchmodels 

clean:
	$(RM) -rf build torchmodels.egg-info *.so dist/ torchmodels/__pycache__/
