.PHONY: dev db test lint

dev:
	python -m uvicorn main:app --reload

db:
	docker compose up -d
	python init_db.py

test:
	pytest -q

lint:
	ruff check .