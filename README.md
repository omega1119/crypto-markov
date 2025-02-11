# Trading Strategy Backtester

## Overview
This project implements a **Markov-based trading strategy backtester** for analyzing Bitcoin (BTC) price movements using historical data. The script:

- Loads and processes trading data from Parquet files.
- Computes price velocity and assigns market states.
- Builds a **Markov transition matrix** to predict future states.
- Simulates trading decisions based on predicted states.
- Evaluates performance using **Return on Investment (ROI)** and **Maximum Drawdown**.
- Provides visualizations of trading performance and drawdowns.

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
Make sure you have **Python 3.8+** installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `networkx`
- `tqdm`
- `pyarrow`
- `notebook`
- `jupyter`

## Usage

### Running the Script
Execute the script using:
```bash
python main.py
```

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
```

## Contribution
Feel free to improve the strategy by modifying the **Markov model**, experimenting with **different indicators**, or **enhancing visualizations**.

## License
MIT License

