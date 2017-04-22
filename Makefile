PYTHON   := venv/bin/python2.7
PYLINT   := pylint
AUTOPEP8 := autopep8

test:
	$(PYTHON) download.py
	$(PYTHON) collaborative-filtering.py data/ratings.csv

download-data:
	rm -rf data
	wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
	wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
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
