# Multiobjective-portfolio-optimization
Python research project comparing NSGA-II and MOLA for multi-objective portfolio optimization, using XGBoost, Optuna, and Ledoit–Wolf covariance estimation.

# MOPO-H | Hybrid Multi-Objective Portfolio Optimization

This repository contains the Python code for my undergraduate research at the Federal University of Itajubá (UNIFEI), supported by FAPEMIG. The project investigates how the Multi-Objective Lichtenberg Algorithm (MOLA) performs in portfolio optimization, using NSGA-II as a benchmark.

The workflow combines next-trading-day return forecasts from XGBoost with Ledoit–Wolf covariance estimation. It then uses those inputs to compare portfolios across three objectives: maximizing expected annualized return, minimizing annualized volatility, and minimizing the number of assets with portfolio weights strictly above 5%.

## Workflow

| Script | What it does |
| --- | --- |
| `etapa1.py` | Downloads historical prices for 11 assets listed on B3 and the Ibovespa index. Computes log returns and technical indicators, tunes one XGBoost model per asset with Optuna using time-series cross-validation, and estimates the covariance matrix with Ledoit–Wolf. Saves the inputs for portfolio optimization to `dados_preparados.pkl`. |
| `etapa2.py` | Loads the prepared inputs and runs NSGA-II and MOLA across ten seeds. Compares normalized hypervolume using the Mann–Whitney U test, combines non-dominated solutions into a global Pareto front, and applies TOPSIS to select a compromise portfolio. |

## How to Run

Install the Python dependencies:

```bash
python -m pip install numpy pandas yfinance scikit-learn xgboost optuna matplotlib pymoo scipy
```

Stage 2 also requires the MOLA implementation that provides `multi_objective_lichtenberg_algorithm`. Install that implementation in the same Python environment before running `etapa2.py`.

Run the scripts in order from the repository directory:

```bash
python etapa1.py
python etapa2.py
```

Stage 1 downloads price data from Yahoo Finance, so it requires an internet connection.

## Outputs

- **Stage 1** generates `dados_preparados.pkl`, model evaluation metrics in `avaliacao_xgboost_optuna.csv`, and the selected hyperparameters in `melhores_parametros_xgboost.csv`.
- **Stage 2** exports run logs, hypervolume results, Pareto-front data and plots, and the portfolio selected by TOPSIS. These outputs are saved in `graficos_resultados_7pct/`. The threshold used by the current script is 5%.
