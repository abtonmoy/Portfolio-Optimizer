# Portfolio-Optimizer

## Overview

This project provides a modular and extensible Python API class called `PriceHistory` for retrieving historical financial data from the Alpha Vantage API. Additionally, the project integrates portfolio optimization techniques such as Monte Carlo simulation and `scipy.optimize` using SLSQP to derive optimal asset allocations based on user-defined metrics like Sharpe Ratio, volatility, and expected returns.

---

## Features

### PriceHistory Class

- **Data Fetching**:

  - Pulls historical stock/ETF data from Alpha Vantage.
  - Handles rate limits and API exceptions.

- **Symbol Management**:

  - Accepts flat lists or nested dictionaries of ticker symbols.

- **Data Cleaning**:

  - Standardizes column names, formats dates, converts data types.
  - Adds features like daily returns, price ranges, and average prices.

- **Convenience Methods**:

  - `build_URL`: Builds the API URL for a given symbol.
  - `symbols`: Returns the stored symbols by category or subcategory.
  - `fetch_prices`: Fetches and processes historical prices.
  - `build_df`: Constructs a unified DataFrame.
  - `get_data_summary`: Summarizes collected data.

---

## Portfolio Optimization

### Monte Carlo Simulation

- Generates thousands of random portfolio weight combinations.
- Calculates:

  - Portfolio return
  - Portfolio volatility
  - Sharpe Ratio

- Visualizes results with a scatter plot colored by Sharpe Ratio.
- Identifies:

  - Portfolio with maximum Sharpe Ratio
  - Portfolio with minimum volatility

### SLSQP Optimization (Scipy)

- Uses Sequential Least Squares Programming (SLSQP) to:

  - Maximize Sharpe Ratio
  - Minimize volatility

- Subject to constraints:

  - Weights must sum to 1
  - No short selling (weights between 0 and 1)

---

## Usage Example

```python
from price_history import PriceHistory

api_key = 'YOUR_API_KEY'
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

ph = PriceHistory(api_key, symbols=symbols)
data = ph.build_df()
summary = ph.get_data_summary()
```

### Optimization Example

```python
from optimizer import monte_carlo_simulation, optimize_portfolio

# Run Monte Carlo simulation
simulation_df = monte_carlo_simulation(data)

# Perform SLSQP optimization
optimal_weights = optimize_portfolio(data)
```

---

## Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `requests`
- `scipy`

Install via pip:

```bash
pip install -r requirements.txt
```

---

## Visualization Outputs

- **Efficient Frontier**
- **Capital Allocation Line (CAL)**
- **Optimal Portfolios Highlighted**

---

## Project Structure

```
.
├── data
│   └── stocks.csv
├── keys
│   └── all_keys.py
├── operations
│   ├── __init__.py
│   └── fetch_data.py
├── optimizations
│   ├── monte_carlo_simulation.ipynb
│   └── scipy_SLSQP_optimization.ipynb
├── venv
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## License

MIT License

---

## Author

**Abdul Basit Tonmoy**
Wabash College

---

## Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for financial market data.
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/) for core scientific computing and visualization.
- sigmacoding
