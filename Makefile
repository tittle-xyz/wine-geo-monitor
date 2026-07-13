.PHONY: install run test lint chart cost-curve advise-chart clean

install:  ## editable install with dev + viz extras
	pip install -e ".[dev,viz]"

run:  ## run the monitor with the mock provider (no API key)
	python -m wine_geo --provider mock --n 30 --seed 42

test:  ## pytest with coverage
	pytest --cov=wine_geo --cov-report=term-missing

lint:  ## ruff
	ruff check .

chart:  ## render the example share-of-voice chart
	python -m wine_geo --provider mock --n 50 --seed 42 --chart out/chart.png --chart-prompt p0

cost-curve:  ## render the cost-of-confidence curve (docs/cost_of_confidence.png)
	python -m wine_geo --provider mock --seed 42 --cost-curve docs/cost_of_confidence.png

advise-chart:  ## render the sample-size planning chart (docs/sample_size.png)
	python -m wine_geo.advise --chart docs/sample_size.png

clean:
	rm -rf out .pytest_cache .ruff_cache .coverage
