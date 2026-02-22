"""
Test which series_ticker values actually return markets from the Kalshi API.
Run with: python debug_series.py
"""

from kalshi_client import KalshiClient

client = KalshiClient()

candidates = [
    # Bitcoin
    "KXBTCD",
    # S&P 500
    "INXD", "KXINXD", "INXPOS", "KXINXPOS",
    # Nasdaq
    "NASDAQ100", "KXNASDAQ100", "NASDAQ100POS", "KXNASDAQ100POS",
    # Fed / interest rates
    "FED", "KXFED", "FEDMEET", "KXFEDMEET",
    # CPI
    "CPI", "KXCPI",
    # Temperature / weather
    "HIGHNY", "KXHIGHNY", "GTEMP", "KXGTEMP", "HMONTH", "KXHMONTH",
    # Crypto misc
    "KXETH", "KXSOL",
]

print(f"{'SERIES TICKER':<30} {'MARKETS FOUND':>14}")
print("-" * 46)

for series in candidates:
    try:
        data = client._get("/markets", params={"limit": 1, "status": "open", "series_ticker": series})
        # Use cursor page count or just check if any returned
        count_data = client._get("/markets", params={"limit": 200, "status": "open", "series_ticker": series})
        count = len(count_data.get("markets", []))
        label = f"{count}+" if count_data.get("cursor") else str(count)
        print(f"{series:<30} {label:>14}")
    except Exception as e:
        print(f"{series:<30} {'ERROR':>14}  ({e})")
