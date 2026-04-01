# Contributing to LongTracer

Thanks for your interest in contributing. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/ENDEVSOLS/LongTracer.git
cd LongTracer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pip install ruff mypy pytest pytest-cov hypothesis
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=longtracer --cov-report=term-missing

# Specific test file
pytest tests/test_verifier.py -v
```

## Code Style

We use ruff for linting and formatting:

```bash
ruff check .
ruff format .
```

Type checking with mypy:

```bash
mypy longtracer/
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Run `ruff check .` and `mypy longtracer/` — both must pass
4. Run `pytest` — all tests must pass
5. Update CHANGELOG.md if applicable
6. Open a PR with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub with:
- Python version and OS
- LongTracer version (`pip show longtracer`)
- Minimal reproduction steps
- Expected vs actual behavior
