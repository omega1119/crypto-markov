from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


LOGGER = logging.getLogger("crypto_markov")


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUTS_DIR = BASE_DIR / "outputs"


REQUIRED_COLUMNS = {
    "Open Time",
    "Close",
}


State = Literal["High Up", "Low Up", "High Down", "Low Down", "Neutral"]


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance: float = 1000.0
    fee: float = 0.001
    trade_allocation: float = 0.2
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    velocity_window: int = 3
    high_quantile: float = 0.75
    low_quantile: float = 0.25
    seed: int = 7
    save_outputs: bool = True
    show_plots: bool = False


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_input_path(
    *,
    data_dir: Path,
    timestamp: int,
    interval: str,
    symbol: str,
    prefer: Literal["parquet", "csv"] = "parquet",
) -> Path:
    base = data_dir / str(timestamp) / interval
    parquet_path = base / f"{symbol}.parquet"
    csv_path = base / f"{symbol}.csv"
    if prefer == "parquet":
        return parquet_path if parquet_path.exists() else csv_path
    return csv_path if csv_path.exists() else parquet_path


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Open Time", "Close"]).sort_values(by="Open Time")
    if len(df) < 10:
        raise ValueError(f"Not enough rows after cleaning: {len(df)}")
    return df.reset_index(drop=True)


def compute_velocity(df: pd.DataFrame, *, window: int) -> pd.DataFrame:
    out = df.copy()
    out["velocity"] = (
        out["Close"].pct_change().rolling(window=window, min_periods=1).mean()
    )
    return out


def assign_state(
    df: pd.DataFrame,
    *,
    high_quantile: float,
    low_quantile: float,
) -> pd.DataFrame:
    out = df.copy()
    high_threshold = out["velocity"].quantile(high_quantile)
    low_threshold = out["velocity"].quantile(low_quantile)

    conditions = [
        out["velocity"] > high_threshold,
        (out["velocity"] > 0) & (out["velocity"] <= high_threshold),
        out["velocity"] < low_threshold,
        (out["velocity"] < 0) & (out["velocity"] >= low_threshold),
    ]
    states: list[State] = ["High Up", "Low Up", "High Down", "Low Down"]
    out["state"] = np.select(conditions, states, default="Neutral")
    return out


def build_transition_matrix(states: Iterable[str]) -> pd.DataFrame:
    s = pd.Series(list(states), name="state").astype("string")
    s_next = s.shift(-1).rename("next_state")
    pairs = pd.concat([s, s_next], axis=1).dropna()
    counts = pd.crosstab(pairs["state"], pairs["next_state"]).astype(float)
    counts = counts.reindex(index=sorted(counts.index), columns=sorted(counts.columns), fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_sums, axis=0).fillna(0.0)
    return probs


def predict_next_state(
    current_state: str,
    transition_matrix: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> str:
    if current_state not in transition_matrix.index:
        return "Neutral"
    probs = transition_matrix.loc[current_state]
    p = probs.to_numpy(dtype=float)
    s = float(p.sum())
    if s <= 0:
        return current_state
    p = p / s
    return str(rng.choice(probs.index.to_list(), p=p))


@dataclass
class BacktestResult:
    final_balance: float
    total_profit: float
    roi_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    sharpe: float


def compute_max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak == 0, np.nan, peak)
    return float(np.nanmax(drawdown))


def sharpe_ratio(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(r) / std * np.sqrt(252))


def backtest(
    df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    *,
    config: BacktestConfig,
    rng: np.random.Generator,
) -> tuple[BacktestResult, pd.DataFrame, pd.DataFrame]:
    balance = float(config.initial_balance)
    position = 0.0
    entry_price: float | None = None
    trade_rows: list[dict[str, object]] = []

    times = df["Open Time"].to_numpy()
    prices = df["Close"].to_numpy(dtype=float)
    states = df["state"].astype(str).to_numpy()

    equity_curve = np.zeros(len(df), dtype=float)
    for i in tqdm(range(1, len(df)), desc="Backtesting", leave=False):
        current_state = str(states[i - 1])
        predicted_state = predict_next_state(current_state, transition_matrix, rng=rng)
        price = float(prices[i])

        # Risk exits
        if position > 0 and entry_price is not None:
            if price <= entry_price * (1.0 - config.stop_loss_pct):
                balance += position * price * (1.0 - config.fee)
                trade_rows.append(
                    {
                        "time": times[i],
                        "price": price,
                        "action": "sell",
                        "reason": "stop_loss",
                    }
                )
                position = 0.0
                entry_price = None
            elif price >= entry_price * (1.0 + config.take_profit_pct):
                balance += position * price * (1.0 - config.fee)
                trade_rows.append(
                    {
                        "time": times[i],
                        "price": price,
                        "action": "sell",
                        "reason": "take_profit",
                    }
                )
                position = 0.0
                entry_price = None

        # Signal-driven entries/exits
        if position == 0.0 and predicted_state in {"High Up", "Low Up"}:
            trade_amount = balance * float(config.trade_allocation)
            if trade_amount > 0:
                position = (trade_amount / price) * (1.0 - config.fee)
                balance -= trade_amount
                entry_price = price
                trade_rows.append(
                    {
                        "time": times[i],
                        "price": price,
                        "action": "buy",
                        "reason": "signal_up",
                    }
                )
        elif position > 0.0 and predicted_state in {"High Down", "Low Down"}:
            balance += position * price * (1.0 - config.fee)
            trade_rows.append(
                {
                    "time": times[i],
                    "price": price,
                    "action": "sell",
                    "reason": "signal_down",
                }
            )
            position = 0.0
            entry_price = None

        equity_curve[i] = balance + position * price

    # Ensure equity at index 0 is sensible
    equity_curve[0] = config.initial_balance
    final_balance = float(balance + position * float(prices[-1]))
    total_profit = float(final_balance - config.initial_balance)
    roi = float(total_profit / config.initial_balance * 100.0)

    equity_df = pd.DataFrame({"time": df["Open Time"], "equity": equity_curve})
    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0.0)
    max_dd = compute_max_drawdown(equity_curve)

    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty:
        trades_df["time"] = pd.to_datetime(trades_df["time"])

    # Trade-level win rate (pair buy->sell sequentially)
    wins = 0
    closed = 0
    if not trades_df.empty:
        stack: list[float] = []
        for _, row in trades_df.iterrows():
            if row["action"] == "buy":
                stack.append(float(row["price"]))
            elif row["action"] == "sell" and stack:
                entry = stack.pop(0)
                closed += 1
                if float(row["price"]) > entry:
                    wins += 1
    win_rate = float((wins / closed) * 100.0) if closed > 0 else 0.0

    result = BacktestResult(
        final_balance=final_balance,
        total_profit=total_profit,
        roi_pct=roi,
        max_drawdown_pct=float(max_dd * 100.0),
        total_trades=int(len(trades_df)),
        win_rate_pct=win_rate,
        sharpe=sharpe_ratio(equity_df["returns"].to_numpy(dtype=float)),
    )
    return result, trades_df, equity_df


def plot_trades(df: pd.DataFrame, trades_df: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Open Time"], df["Close"], label="Close Price")
    if not trades_df.empty:
        buys = trades_df[trades_df["action"] == "buy"]
        sells = trades_df[trades_df["action"] == "sell"]
        ax.scatter(buys["time"], buys["price"], color="green", marker="^", label="Buy", alpha=0.8)
        ax.scatter(sells["time"], sells["price"], color="red", marker="v", label="Sell", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_equity(equity_df: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    equity = equity_df["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak == 0, np.nan, peak)
    dd = np.nan_to_num(dd, nan=0.0)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(equity_df["time"], equity_df["equity"], label="Equity")
    ax1.set_title("Equity Curve")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("USDT")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(equity_df["time"], dd * 100.0, label="Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("%")
    ax2.legend()
    fig2.tight_layout()

    return fig1, fig2


def plot_transition_heatmap(transition_matrix: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Markov Transition Matrix")
    fig.tight_layout()
    return fig


def plot_transition_network(transition_matrix: pd.DataFrame, *, threshold: float = 0.05) -> plt.Figure:
    g = nx.DiGraph()
    for src in transition_matrix.index:
        for dst in transition_matrix.columns:
            w = float(transition_matrix.loc[src, dst])
            if w >= threshold:
                g.add_edge(str(src), str(dst), weight=w)

    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(g, seed=7)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1200)
    nx.draw_networkx_labels(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowstyle="-|>")
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title("State Transition Network")
    ax.axis("off")
    fig.tight_layout()
    return fig


def save_outputs(
    *,
    outputs_dir: Path,
    result: BacktestResult,
    config: BacktestConfig,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    figures: dict[str, plt.Figure],
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "metrics.json").write_text(
        json.dumps({"config": asdict(config), "result": asdict(result)}, indent=2),
        encoding="utf-8",
    )
    trades_df.to_csv(outputs_dir / "trades.csv", index=False)
    equity_df.to_csv(outputs_dir / "equity.csv", index=False)
    transition_matrix.to_csv(outputs_dir / "transition_matrix.csv")
    for name, fig in figures.items():
        fig.savefig(outputs_dir / f"{name}.png", dpi=150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markov-based trading strategy backtester")
    parser.add_argument("--input", type=str, default="", help="Path to .parquet or .csv")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--timestamp", type=int, default=1738439134000)
    parser.add_argument("--interval", type=str, default="30m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--prefer", choices=["parquet", "csv"], default="parquet")

    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--trade-allocation", type=float, default=0.2)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.04)

    parser.add_argument("--velocity-window", type=int, default=3)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--no-save", action="store_true", help="Do not write outputs/ files")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_path = Path(args.input) if args.input else resolve_input_path(
        data_dir=Path(args.data_dir),
        timestamp=args.timestamp,
        interval=args.interval,
        symbol=args.symbol,
        prefer=args.prefer,
    )
    input_path = input_path.expanduser().resolve() if args.input else input_path
    LOGGER.info("Loading data: %s", input_path)
    df = load_data(input_path)

    config = BacktestConfig(
        initial_balance=args.initial_balance,
        fee=args.fee,
        trade_allocation=args.trade_allocation,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        velocity_window=args.velocity_window,
        high_quantile=args.high_quantile,
        low_quantile=args.low_quantile,
        seed=args.seed,
        save_outputs=not args.no_save,
        show_plots=bool(args.show),
    )

    df = compute_velocity(df, window=config.velocity_window)
    df = assign_state(df, high_quantile=config.high_quantile, low_quantile=config.low_quantile)
    transition_matrix = build_transition_matrix(df["state"].values)

    rng = np.random.default_rng(config.seed)
    result, trades_df, equity_df = backtest(df, transition_matrix, config=config, rng=rng)

    print(f"Final Balance: {result.final_balance:.2f} USDT")
    print(f"Total Profit: {result.total_profit:.2f} USDT")
    print(f"ROI: {result.roi_pct:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate_pct:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe (daily scaled): {result.sharpe:.2f}")

    figures: dict[str, plt.Figure] = {}
    figures["trades"] = plot_trades(df, trades_df, title="Trading Backtest Results")
    eq_fig, dd_fig = plot_equity(equity_df)
    figures["equity"] = eq_fig
    figures["drawdown"] = dd_fig
    figures["transition_heatmap"] = plot_transition_heatmap(transition_matrix)
    figures["transition_network"] = plot_transition_network(transition_matrix)

    if config.save_outputs:
        save_outputs(
            outputs_dir=Path(args.outputs_dir),
            result=result,
            config=config,
            trades_df=trades_df,
            equity_df=equity_df,
            transition_matrix=transition_matrix,
            figures=figures,
        )
        LOGGER.info("Wrote outputs to: %s", Path(args.outputs_dir).resolve())

    if config.show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()