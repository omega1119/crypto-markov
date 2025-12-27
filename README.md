# Trading Strategy Backtester

## Overview
This project implements a **Markov-based trading strategy backtester** for analyzing Bitcoin (BTC) price movements using historical data. The script:

- Loads and processes trading data from Parquet files.
- Computes price velocity and assigns market states.
- Builds a **Markov transition matrix** to predict future states.
- Simulates trading decisions based on predicted states.
- Simulates trading decisions based on predicted states with optional **stop-loss** and **take-profit**.
- Evaluates performance using **ROI**, **Max Drawdown**, **Win Rate**, and **Sharpe (daily scaled)**.
- Produces visualizations (trades, equity curve, drawdown, transition matrix heatmap, transition network).

Additionally, a **Jupyter Notebook** is included to provide a detailed breakdown of the `main.py` script. The notebook explains each section of the code, including the mathematical and theoretical concepts behind the Markov-based trading strategy. It is an excellent resource for understanding the methodology and implementation.

## Features
- **Data Handling**: Reads BTC price data from Parquet files.
- **Markov Chain Analysis**: Predicts market state transitions.
- **Backtesting**: Simulates trading based on market state predictions.
- **Risk Management**: Tracks **Max Drawdown** and **ROI**.
- **Visualizations**: Displays trading performance and risk metrics.
- **Detailed Jupyter Notebook**: Explains the trading strategy with step-by-step markdown and equations.

## Installation

### Requirements
Make sure you have **Python 3.11+** installed.

### Virtual Environment (venv) Setup (macOS)
From the project root:

1) Verify Python version:
```bash
python3 --version
```

2) Create a virtual environment (recommended name: `.venv`):
```bash
python3 -m venv .venv
```

3) Activate it:
```bash
source .venv/bin/activate
```

4) Upgrade pip tooling and install dependencies:
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

5) Run the backtest:
```bash
python main.py
```

To deactivate later:
```bash
deactivate
```

### Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
```
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `networkx`
- `tqdm`
- `pyarrow`

The notebook is optional; install Jupyter separately if you want to run it.

## Usage

### Running the Script
Execute the script using:
```bash
python main.py
```

### Common CLI Options
- Run on a specific file:
	- `python main.py --input data/1738439134000/30m/BTCUSDT.parquet`
- Or resolve from parts:
	- `python main.py --timestamp 1738439134000 --interval 30m --symbol BTCUSDT`
- Reproducible runs:
	- `python main.py --seed 7`
- Adjust strategy/risk parameters:
	- `python main.py --velocity-window 3 --high-quantile 0.75 --low-quantile 0.25`
	- `python main.py --trade-allocation 0.2 --stop-loss 0.02 --take-profit 0.04 --fee 0.001`
- Plot behavior:
	- `python main.py --show` (interactive)
	- default: saves plots to `outputs/` and closes figures

Alternatively, open the **Jupyter Notebook** to explore the trading strategy step by step:
```bash
jupyter notebook
```

### Expected Output
The script will output:
- **Final balance** after simulated trades.
- **Total profit** earned.
- **ROI (%)** for the backtest period.
- **Total number of trades executed**.
- **Max Drawdown (%)** as a risk measure.
- **Win Rate (%)** on closed trades.
- **Sharpe ratio** (daily scaled; rough summary metric).

### Output Files
By default the script writes artifacts to `outputs/`:
- `metrics.json`
- `trades.csv`
- `equity.csv`
- `transition_matrix.csv`
- `*.png` plots

### Visualizations
- **Trading Performance Chart**: Buy and sell points on the BTC price chart.
- **Max Drawdown Over Time**: Risk visualization.
- **Markov Transition Matrix Heatmap**: Probability of market state changes.
- **State Transition Network**: Visual representation of Markov state transitions.

## Folder Structure
```
.
├── data/             # Trading data (Parquet format)
├── main.py           # Main script
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
├── notebook.ipynb    # Jupyter Notebook explaining the strategy
├── outputs/          # Generated metrics/plots (created on run)
```

## Contribution
Feel free to improve the strategy by modifying the **Markov model**, experimenting with **different indicators**, or **enhancing visualizations**.

## License
MIT License

