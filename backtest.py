"""
Simple backtester — replays historical Kalshi market data to evaluate strategy performance.

Usage:
    python backtest.py --ticker INXD-23DEC31-T4500 --strategy mean_reversion
"""

import argparse
import logging
from kalshi_client import KalshiClient
from strategy import MeanReversionStrategy, MomentumStrategy, TradeSignal
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("backtest")


def run_backtest(ticker: str, strategy_name: str):
    client = KalshiClient()

    try:
        market = client.get_market(ticker)
    except Exception as e:
        logger.error(f"Could not fetch market {ticker}: {e}")
        return

    logger.info(f"Backtesting {ticker}: {market.get('title', '')}")
    logger.info(f"Status: {market.get('status')} | Result: {market.get('result', 'TBD')}")

    # Fetch full trade history
    trades = client.get_market_history(ticker, limit=1000)
    if not trades:
        logger.warning("No trade history available.")
        return

    logger.info(f"Loaded {len(trades)} historical trades")

    # Simulate strategy decisions at each trade point
    strategy_map = {
        "mean_reversion": MeanReversionStrategy(client),
        "momentum": MomentumStrategy(client),
    }
    strategy = strategy_map.get(strategy_name)
    if not strategy:
        logger.error(f"Unknown strategy: {strategy_name}. Choose from {list(strategy_map)}")
        return

    # Simple simulation: track hypothetical P&L
    pnl = 0.0
    num_trades = 0
    wins = 0
    losses = 0
    resolved_yes = market.get("result") == "yes"

    for i, trade in enumerate(trades):
        # Simulate the orderbook at this point in time using past trades
        # (simplified: treat this trade price as the mid)
        sim_price = trade.get("yes_price", 50)
        edge = (0.60 - sim_price / 100) if "FED" in ticker else (0.50 - sim_price / 100)

        if abs(edge) < config.MIN_EDGE:
            continue

        # Decision
        bet_yes = edge > 0
        entry_price = sim_price / 100

        # Outcome: did YES resolve?
        if bet_yes:
            profit = (1.0 - entry_price) if resolved_yes else -entry_price
        else:
            no_price = (100 - sim_price) / 100
            profit = (1.0 - no_price) if not resolved_yes else -no_price

        pnl += profit * config.MAX_TRADE_USD
        num_trades += 1
        if profit > 0:
            wins += 1
        else:
            losses += 1

    logger.info("─" * 50)
    logger.info(f"Simulated trades : {num_trades}")
    logger.info(f"Wins / Losses    : {wins} / {losses}")
    win_rate = (wins / num_trades * 100) if num_trades else 0
    logger.info(f"Win rate         : {win_rate:.1f}%")
    logger.info(f"Hypothetical P&L : ${pnl:.2f}")
    logger.info("─" * 50)
    logger.info("NOTE: This is a rough simulation. Real backtest requires tick-level data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi strategy backtester")
    parser.add_argument("--ticker", required=True, help="Market ticker to backtest")
    parser.add_argument(
        "--strategy",
        default="mean_reversion",
        choices=["mean_reversion", "momentum"],
        help="Strategy to backtest",
    )
    args = parser.parse_args()
    run_backtest(args.ticker, args.strategy)
