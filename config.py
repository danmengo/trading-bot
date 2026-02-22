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
MAX_OPEN_POSITIONS = 10000  # set to a low number (e.g. 3) to cap concurrent positions
POLL_INTERVAL_SECONDS = 300  # 5 minutes — daily contracts don't reprice faster than this
ORDER_CANCEL_AFTER_MINUTES = 20  # cancel unfilled resting orders after this long

# Specific market tickers to always trade regardless of series filters.
# Add any ticker you want to target directly, e.g. "KXBTCD-26FEB2717-T76499.99"
WATCHED_TICKERS = []

WATCHED_SERIES = [
    "KXBTCD",  # Bitcoin daily
    "KXETH",   # Ethereum daily
]

# Only trade contracts expiring within this many days.
SERIES_MAX_DAYS_TO_EXPIRY = {
    "KXBTCD": 7,
    "KXETH":  7,
}

# ── Crypto price strategy ──────────────────────────────────────────────────────
# Only trade contracts whose threshold is within this % of the current spot price.
# Tighter = fewer but higher-quality near-the-money trades.
CRYPTO_NEAR_MONEY_PCT = 0.08  # 8% from spot price

# Assumed daily volatility per asset (tune based on observed market behaviour).
CRYPTO_DAILY_VOL = {
    "KXBTCD": 0.035,  # BTC: ~3.5% daily vol
    "KXETH":  0.045,  # ETH: ~4.5% daily vol
}

# ── Risk management ───────────────────────────────────────────────────────────
# Stop the bot if you lose more than this in a session (~38% of bankroll)
DAILY_LOSS_LIMIT_USD = 3.00
DRY_RUN = False  # Set to False when ready to trade real money
