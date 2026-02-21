"""
Configuration — all secrets are loaded from .env (never hardcode keys here).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

# ── Kalshi ────────────────────────────────────────────────────────────────────
# The UUID shown under your API key in Kalshi account settings
KALSHI_KEY_ID = os.environ["KALSHI_KEY_ID"]

# The RSA private key PEM — newlines stored as \n in .env
# Kalshi shows this ONCE when you create the key. If you lost it, delete and recreate.
KALSHI_PRIVATE_KEY = os.environ["KALSHI_PRIVATE_KEY"]

KALSHI_ENV = os.getenv("KALSHI_ENV", "prod")

KALSHI_BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}

# ── Gemini (optional) ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # empty = AI sentiment disabled

# ── Trading parameters ────────────────────────────────────────────────────────
# With ~$7.81 balance: keep trades small so you can hold 2-3 positions at once
# and always have a cash buffer. $2.50 × 3 positions = $7.50 max exposure.
MAX_TRADE_USD = 2.50
MIN_EDGE = 0.05
MAX_OPEN_POSITIONS = 3
POLL_INTERVAL_SECONDS = 60

WATCHED_SERIES = [
    "FED-",       # Fed rate decisions (active around FOMC meetings)
    "CPI-",       # CPI reports
    "INXD-",      # S&P 500 daily
    "KXBTCD-",    # Bitcoin daily
    "HIGHNY-",    # NYC daily high temp (always open)
    "NASDAQ100-", # Nasdaq daily
]

# ── Risk management ───────────────────────────────────────────────────────────
# Stop the bot if you lose more than this in a session (~38% of bankroll)
DAILY_LOSS_LIMIT_USD = 3.00
DRY_RUN = False  # Set to False when ready to trade real money
