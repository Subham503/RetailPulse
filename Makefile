.PHONY: setup pipeline clean dashboard api mlflow test docker-build docker-up

# ── Setup ──────────────────────────────────────────────
setup:
	cp -n .env.example .env || true
	pip install -r requirements.txt
	@echo "✅ Setup complete. Drop online_retail_II.xlsx into data/raw/"

# ── Pipeline steps ─────────────────────────────────────
ingest:
	python -m src.data.ingest

clean:
	python -m src.data.clean

features:
	python -m src.features.rfm
	python -m src.features.time_features

models:
	python -m src.models.segmentation
	python -m src.models.forecasting
	python -m src.models.churn
	python -m src.models.inventory

pipeline: ingest clean features models
	@echo "✅ Full pipeline complete"

# ── Serving ────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py --server.port 8501

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Testing ────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Docker ─────────────────────────────────────────────
docker-build:
	docker build -t retailpulse:latest -f docker/Dockerfile .

docker-up:
	docker compose -f docker/docker-compose.yml up

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Cleanup ────────────────────────────────────────────
clean-outputs:
	rm -rf data/processed/* data/features/* models/*.joblib models/*.pkl
	@echo "✅ Outputs cleaned (raw data preserved)"

clean-mlflow:
	rm -rf mlruns/
	@echo "✅ MLflow runs cleared"
