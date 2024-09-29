format:
	@echo "Formatting backend code..."
	poetry run ruff format .

lint:
	@echo "Linting backend code..."
	poetry run ruff check --fix

lint-and-format: format lint