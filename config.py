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
# If the model and market disagree by more than this, skip the trade.
# A huge gap (e.g. model=0.99, market=0.12) means the market has information
# the log-normal model doesn't — momentum, news, order flow. Trust the market.
MAX_MODEL_MARKET_DISAGREEMENT = 0.50
MAX_OPEN_POSITIONS = 5  # set to a low number (e.g. 3) to cap concurrent positions
POLL_INTERVAL_SECONDS = 300  # 5 minutes — daily contracts don't reprice faster than this
ORDER_CANCEL_AFTER_MINUTES = 20  # cancel unfilled resting orders after this long

# Specific market tickers to always trade regardless of series filters.
# Add any ticker you want to target directly, e.g. "KXBTCD-26FEB2717-T76499.99"
WATCHED_TICKERS = []

WATCHED_SERIES = [
    "KXBTCD",  # Bitcoin daily
    "KXETH",   # Ethereum daily
]

# Trading window: only trade contracts within this many hours of expiry.
# The log-normal model is sharpest in the 2–12 hour window before expiry:
#   - Too close (< 2h): model breaks down, stale spot prices cause large errors.
#   - Too far   (> 12h): probabilities compress toward 50%, less signal, overnight risk.
# Run the bot at 7–10 AM PST to catch the 5 PM EST daily expiry at the right horizon.
MIN_HOURS_TO_EXPIRY = 2.0
MAX_HOURS_TO_EXPIRY = 12.0

# ── Crypto price strategy ──────────────────────────────────────────────────────
# Only trade contracts whose threshold is within this % of the current spot price.
# Tighter = fewer but higher-quality near-the-money trades.
CRYPTO_NEAR_MONEY_PCT = 0.08  # 8% from spot price

# Daily volatility per asset — used as fallback if CoinGecko price history is
# unavailable. CryptoPriceStrategy auto-calibrates from 30-day realized vol on
# startup and logs which value it is using.
# Update these if the bot has been offline for a long time and auto-cal fails.
CRYPTO_DAILY_VOL = {
    "KXBTCD": 0.024,  # BTC: ~2.4% daily (38% annualized)
    "KXETH":  0.032,  # ETH: ~3.2% daily (~1.3x BTC)
}

# Skip a contract if the YES bid-ask spread exceeds this many cents.
# A wide spread signals an illiquid market where the mid price is unreliable.
# The edge threshold is also widened by half the spread (see strategy.py).
CRYPTO_MAX_SPREAD_CENTS = 10

# ── Risk management ───────────────────────────────────────────────────────────
# Stop the bot if you lose more than this in a session (~38% of bankroll)
DAILY_LOSS_LIMIT_USD = 3.00
DRY_RUN = False  # Set to False when ready to trade real money

# Exit a position early if it has lost this fraction of its cost.
# e.g. 0.50 = sell when the position is worth less than 50% of what you paid.
# No take-profit — on Kalshi you can only exit at the bid (below mid), so
# winners are better held to expiry at full $1.00 payout.
STOP_LOSS_PCT = 0.50
