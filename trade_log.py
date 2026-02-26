"""
Trade logger — appends one row per order to trades.csv.

Columns:
  timestamp, ticker, strategy, side, action, contracts,
  limit_price_cents, cost_usd, edge, order_id, dry_run
"""

import csv
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(__file__), "trades.csv")

FIELDS = [
    "timestamp",
    "ticker",
    "strategy",
    "side",
    "action",
    "contracts",
    "limit_price_cents",
    "cost_usd",
    "edge",
    "order_id",
    "dry_run",
    "reason",
]


def _ensure_header():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def log_trade(signal, strategy_name: str, order_result: dict, dry_run: bool):
    """Append a single trade row to trades.csv."""
    _ensure_header()
    cost = (signal.limit_price / 100) * signal.contracts
    order_id = (
        order_result.get("order", {}).get("order_id")
        or order_result.get("order_id")
        or ""
    )
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": signal.ticker,
        "strategy": strategy_name,
        "side": signal.side,
        "action": signal.action,
        "contracts": signal.contracts,
        "limit_price_cents": signal.limit_price,
        "cost_usd": f"{cost:.2f}",
        "edge": f"{signal.edge:.4f}",
        "order_id": order_id,
        "dry_run": dry_run,
        "reason": signal.reason,
    }
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
