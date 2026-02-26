"""
Diagnostic script — verifies that get_fills, get_settlements, and
refresh_realized_pnl() are working correctly.

Run with:
    python debug_pnl.py

Cross-check the output against your Kalshi account history page.
"""

from datetime import datetime, timezone
from kalshi_client import KalshiClient
from risk import RiskManager

client = KalshiClient()
risk = RiskManager(client)

today_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
min_ts = int(today_midnight.timestamp())

print(f"\n── Fills since {today_midnight.strftime('%Y-%m-%d %H:%M UTC')} ──")
fills = client.get_fills(min_ts=min_ts)
if not fills:
    print("  (none)")
for f in fills:
    side = f.get("side", "?")
    yes_price = f.get("yes_price", 0)
    price = yes_price if side == "yes" else (100 - yes_price)
    cost = f.get("count", 0) * price / 100
    print(
        f"  {f.get('ticker')} | {f.get('action')} {f.get('count')}x {side.upper()} "
        f"@ {price}c = ${cost:.2f}"
    )

print(f"\n── Settlements since {today_midnight.strftime('%Y-%m-%d %H:%M UTC')} ──")
settlements = client.get_settlements(min_ts=min_ts)
if not settlements:
    print("  (none)")
for s in settlements:
    revenue = s.get("revenue", 0) / 100
    print(f"  {s.get('ticker')} | revenue=${revenue:.2f} | result={s.get('market_result', '?')}")

print("\n── Realized P&L calculation ──")
risk.refresh_realized_pnl()
print(f"  _realized_pnl_today = ${risk._realized_pnl_today:.2f}")
print("\nCross-check this against: Kalshi account → Portfolio → History")
