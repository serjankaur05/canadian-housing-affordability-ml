# ds3000-housing-burden

Private DataSci 3000 term project workspace for housing burden analysis.

## Recommended workflow

- Use Cursor for editing/debugging code in `src/`.
- Use Colab for shared notebook execution and outputs in `notebooks/`.
- Use GitHub (private repo) as source of truth and collaboration.

## Project structure

```text
ds3000-housing-burden/
├── data/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
├── notebooks/
│   └── DataSciTermProject.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── data_preparation.py
│   ├── plotting.py
│   ├── supervised.py
│   └── unsupervised.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Data policy

- Keep this repository private.
- Do not commit raw course-provided confidential datasets.
- Keep files in local `data/` only.

Expected local files:

- `data/DemoStats_(DST)_2020_2025_GEO.csv`
- `data/HouseholdSpend_2024_GEO.csv`

## Notebook path usage

From `notebooks/`, use:

```python
DEMO_PATH = "../data/DemoStats_(DST)_2020_2025_GEO.csv"
SPEND_PATH = "../data/HouseholdSpend_2024_GEO.csv"
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Work primarily in `src/`.
4. Keep notebook cells focused on orchestration, plots, and report-ready outputs.
