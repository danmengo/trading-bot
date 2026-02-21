# Kalshi Trading Bot

A modular prediction market trading bot for [Kalshi](https://kalshi.com).

## Quick Start

```bash
pip install -r requirements.txt
```

Edit `config.py`:
- Set `KALSHI_API_KEY` (get from https://kalshi.com/account/api)
- Set `KALSHI_ENV = "demo"` to start safely
- Optionally set `GEMINI_API_KEY` for AI sentiment (https://aistudio.google.com/app/apikey)
- Keep `DRY_RUN = True` until you trust your strategy

Run the bot:
```bash
python bot.py
```

Backtest a strategy on a past market:
```bash
python backtest.py --ticker INXD-23DEC31-T4500 --strategy mean_reversion
```

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings and API keys |
| `kalshi_client.py` | Kalshi REST API wrapper |
| `strategy.py` | Trading strategies (mean reversion, momentum, Gemini AI) |
| `risk.py` | Risk management (position limits, loss limits) |
| `bot.py` | Main loop |
| `backtest.py` | Historical simulation |

## Strategies

### MeanReversionStrategy
Set prior probabilities in `strategy.py` → `PRIORS` dict.
Trades when market price deviates from your prior by more than `MIN_EDGE`.

### MomentumStrategy
Follows recent price trends. Good during fast-moving events.

### GeminiSentimentStrategy
Asks Gemini to estimate YES probability from market title.
Only runs if `GEMINI_API_KEY` is set. Uses `gemini-1.5-flash` (free tier).

## Safety

- `DRY_RUN = True` — never submits real orders, just logs them
- `DAILY_LOSS_LIMIT_USD` — bot stops trading if losses exceed this
- `MAX_OPEN_POSITIONS` — caps concurrent exposure
- `MAX_TRADE_USD` — caps per-trade size

## Recommended Niche

**Economic indicators** (FED rate decisions, CPI) are the best starting niche:
- Clear resolution criteria
- Correlated with public data (CME FedWatch, BLS releases)
- Less noise than political markets
- You can build a simple prior from historical data
