.PHONY: install run test lint chart clean

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

clean:
	rm -rf out .pytest_cache .ruff_cache .coverage
