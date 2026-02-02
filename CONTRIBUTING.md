# Contributing

Thanks for your interest in this project. It's a portfolio demo, but contributions and feedback are welcome.

## Local Development Setup

1. Clone and create a virtual environment:
```bash
git clone https://github.com/nburke32/CRE-Sample-Dashboard.git
cd streamlit-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set up API credentials in `.env`:
```bash
FRED_API_KEY=your_fred_api_key
NYC_OPENDATA_APP_TOKEN=your_nyc_token
```

3. Run the app:
```bash
streamlit run Home.py
```

## Running Tests

```bash
# All tests (skip slow/API-dependent)
python -m pytest tests/ -v -m "not slow"

# Include slow tests (requires API keys)
python -m pytest tests/ -v
```

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting. Configuration is in `ruff.toml`.

```bash
# Check for issues
ruff check .

# Auto-fix what can be fixed
ruff check --fix .
```

## Branch Strategy

- **`main`** — Production branch. Streamlit Community Cloud deploys from here. Protected; requires PR.
- **`uat`** — Development/staging branch. Push feature work here first.

## Pull Request Process

1. Create a feature branch from `uat`
2. Make your changes
3. Ensure `ruff check .` passes
4. Ensure `pytest tests/ -m "not slow"` passes
5. Open a PR to `uat`
6. After review and merge to `uat`, a separate PR promotes `uat` to `main`

## Code Style

- Python 3.11+ type hints (`str | None` not `Optional[str]`)
- Imports sorted by isort rules (stdlib, third-party, local)
- Max line length: 120 characters
- Descriptive variable names; no single-letter variables outside loops
