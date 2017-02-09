PYTHON   := python3.5
PIP      := pip3.5
PYLINT   := pylint
AUTOPEP8 := autopep8

test:
	make dependencies
	python3.5 collaborative-filtering.py data/ratings.csv

dependencies:
	rm -rf data
	wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
	unzip ml-latest-small.zip
	mv ml-latest-small data
	rm -rf ml-latest-small.zip

figures:
	rm -rf data figures
	make dependencies
	mkdir figures
	$(PYTHON) ratings-count-distribution.py ./data/ratings.csv ./figures/ratings-count-distribution.png

clean:
	rm -rf data figures
