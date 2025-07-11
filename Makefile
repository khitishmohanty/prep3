MYPY_OPTIONS = --config-file pyproject.toml

.PHONY: test
test:
	poetry run pytest tests

.PHONY: lint-check
lint-check:
	poetry run flake8 src tests

.PHONY: type-check
type-check:
	poetry run mypy ${MYPY_OPTIONS} src tests

.PHONY: install
install:
	poetry install

test:
	poetry run pytest --html=reports/testreports/test_report.html --self-contained-html

coverage:
	poetry run pytest --cov=political_party_analysis --cov-report=html:reports/codecoverage
