import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and Constants
SCRIPT_DIR = Path.cwd()
DATA_FOLDER = SCRIPT_DIR / "data"
MODEL_FOLDER = SCRIPT_DIR / "models"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

TIME_STAMP = 1738439134000
KLINE_INTERVAL = "30m"
SYMBOL = "BTCUSDT"

PARQUET_PATH = str(DATA_FOLDER / f"{TIME_STAMP}/{KLINE_INTERVAL}/{SYMBOL}.parquet")

# Load CSV Data with validation
def load_data(file_path):
    if not Path(file_path).exists():
        logging.error("Data file does not exist: %s", file_path)
        return None
    df = pd.read_parquet(file_path)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df = df.sort_values(by='Open Time')
    return df.dropna()

# Compute velocity with optimized rolling window
def compute_velocity(df):
    df['velocity'] = df['Close'].pct_change().rolling(window=3, min_periods=1).mean()
    return df

# Assign states based on velocity thresholds
def assign_state(df):
    high_threshold = df['velocity'].quantile(0.75)
    low_threshold = df['velocity'].quantile(0.25)
    
    conditions = [
        df['velocity'] > high_threshold,
        (df['velocity'] > 0) & (df['velocity'] <= high_threshold),
        df['velocity'] < low_threshold,
        (df['velocity'] < 0) & (df['velocity'] >= low_threshold)
    ]
    states = ['High Up', 'Low Up', 'High Down', 'Low Down']
    
    df['state'] = np.select(conditions, states, default='Neutral')
    return df

# Build Markov Transition Matrix
def build_transition_matrix(states):
    unique_states = list(set(states))
    transition_counts = defaultdict(lambda: defaultdict(int))

    for (state1, state2) in zip(states[:-1], states[1:]):
        transition_counts[state1][state2] += 1

    transition_matrix = {state: {s: 0 for s in unique_states} for state in unique_states}
    
    for state, transitions in transition_counts.items():
        total = sum(transitions.values())
        for s, count in transitions.items():
            transition_matrix[state][s] = count / total if total > 0 else 0

    return transition_matrix

# Predict Next State
def predict_next_state(current_state, transition_matrix):
    if current_state not in transition_matrix:
        return 'Neutral'  # Fallback strategy
    next_states = list(transition_matrix[current_state].keys())
    probabilities = list(transition_matrix[current_state].values())
    return np.random.choice(next_states, p=probabilities)

# Backtesting Simulation with Portfolio Diversification and Risk Metrics
def backtest(df, transition_matrix, initial_balance=1000, fee=0.001, trade_allocation=0.2, risk_tolerance=0.02):
    balance = initial_balance  # USDT balance
    position = 0  # BTC holdings
    trade_history = []  # Store trade records
    max_drawdown = 0
    peak_balance = initial_balance

    for i in tqdm(range(1, len(df)), desc="Backtesting Progress"):
        current_state = df.iloc[i - 1]['state']
        predicted_state = predict_next_state(current_state, transition_matrix)
        price = df.iloc[i]['Close']

        if position > 0:  # Check stop loss and take profit
            value = position * price
            if value > peak_balance:
                peak_balance = value
            drawdown = (peak_balance - value) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)

        if predicted_state == "Neutral":
            continue

        if position == 0 and predicted_state in ['High Up', 'Low Up']:
            trade_amount = balance * trade_allocation * (1 - risk_tolerance)
            if trade_amount > 0:
                position = trade_amount / price * (1 - fee)
                balance -= trade_amount
                trade_history.append((df.iloc[i]['Open Time'], price, 'buy'))

        elif position > 0 and predicted_state in ['High Down', 'Low Down']:
            balance += position * price * (1 - fee)
            position = 0
            trade_history.append((df.iloc[i]['Open Time'], price, 'sell'))
    
    final_balance = balance + (position * df.iloc[-1]['Close'])
    total_profit = final_balance - initial_balance
    roi = (total_profit / initial_balance) * 100
    
    return final_balance, trade_history, max_drawdown, total_profit, roi

# Improved Visualization
def plot_trades(df, trade_history):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Open Time'], df['Close'], label='Close Price', color='blue')
    trade_df = pd.DataFrame(trade_history, columns=['Time', 'Price', 'Action'])
    buy_trades = trade_df[trade_df['Action'] == 'buy']
    sell_trades = trade_df[trade_df['Action'] == 'sell']
    plt.scatter(buy_trades['Time'], buy_trades['Price'], color='green', marker='^', label='Buy', alpha=0.8)
    plt.scatter(sell_trades['Time'], sell_trades['Price'], color='red', marker='v', label='Sell', alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Trading Backtest Results')
    plt.legend()
    plt.show()

# Run Analysis
def main():
    df = load_data(PARQUET_PATH)
    if df is None:
        return
    df = compute_velocity(df)
    df = assign_state(df)
    
    transition_matrix = build_transition_matrix(df['state'].values)
    final_balance, trade_history, max_drawdown, total_profit, roi = backtest(df, transition_matrix)

    print(f"Final Balance: {final_balance:.2f} USDT")
    print(f"Total Profit: {total_profit:.2f} USDT")
    print(f"ROI: {roi:.2f}%")
    print(f"Total Trades: {len(trade_history)}")
    
    logging.info(f"Max Drawdown: {max_drawdown:.2%}")
    
    plot_trades(df, trade_history)

if __name__ == "__main__":
    main()