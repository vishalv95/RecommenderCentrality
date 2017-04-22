PYTHON   := venv/bin/python2.7

test:
	$(PYTHON) download.py
	$(PYTHON) collaborative-filtering.py data/ratings.csv

clean:
	rm -rf data
