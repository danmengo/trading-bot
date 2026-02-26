"""
Prints raw portfolio position data from Kalshi so you can verify
which fields are available for stop-loss calculations.

Run with: python debug_positions.py
"""

from kalshi_client import KalshiClient
import config

client = KalshiClient()
positions = client.get_portfolio()

print(f"\n── Portfolio ({len(positions)} positions) ──")
if not positions:
    print("  (none)")

for pos in positions:
    ticker = pos.get("ticker", "?")
    print(f"\n  {ticker}")
    for key, val in pos.items():
        print(f"    {key}: {val}")

    # Show what the stop-loss code now sees
    count = pos.get("position", 0)
    cost_cents = pos.get("market_exposure", 0)
    side = "yes" if count > 0 else "no"

    print(f"\n  → stop-loss inputs:")
    print(f"    position        = {count}  ({'active' if count != 0 else 'closed/skip'})")
    print(f"    market_exposure = {cost_cents}c  (${cost_cents/100:.2f} cost basis)")
    print(f"    side            = {side}")

    if count == 0:
        print(f"    SKIP — position is 0 (closed or historical)")
    elif cost_cents <= 0:
        print(f"    SKIP — no cost basis")
    else:
        orderbook = client.get_orderbook(ticker)
        bid = client.best_yes_bid(orderbook) if side == "yes" else client.best_no_bid(orderbook)
        if bid:
            current_value = bid * abs(count)
            loss_pct = (cost_cents - current_value) / cost_cents
            print(f"    best_{side}_bid  = {bid}c")
            print(f"    current_value   = {current_value}c  (${current_value/100:.2f})")
            print(f"    loss_pct        = {loss_pct:.0%}  (threshold={config.STOP_LOSS_PCT:.0%})")
            if loss_pct >= config.STOP_LOSS_PCT:
                print(f"    *** WOULD TRIGGER stop-loss — sell {abs(count)}x {side.upper()} at {bid}c ***")
            else:
                print(f"    below threshold — no stop-loss")
        else:
            print(f"    WARNING: no {side} bid available in orderbook")
