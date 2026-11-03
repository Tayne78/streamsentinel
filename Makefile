.PHONY: install test lint format up up-sim down logs train clean

install:
	pip install -e ".[all,spark]"

test:
	pytest -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

up:            ## live Binance data
	docker compose up -d --build
	@echo "Dashboard: http://localhost:$${API_PORT:-8000}   MLflow: http://localhost:$${MLFLOW_PORT:-5001}"

up-sim:        ## simulated market with labelled anomalies (live precision/recall on the dashboard)
	SOURCE=simulator docker compose up -d --build
	@echo "Dashboard: http://localhost:$${API_PORT:-8000}   MLflow: http://localhost:$${MLFLOW_PORT:-5001}"

down:
	docker compose down

logs:
	docker compose logs -f detector producer spark

train:         ## retrain; the detector hot-swaps the new champion within a minute
	docker compose run --rm trainer

clean:
	docker compose down -v
