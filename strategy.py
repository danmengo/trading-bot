"""
Trading strategy engine.

Core idea (mean-reversion / fair-value):
  1. Estimate the "true" probability of the event using external data or priors.
  2. Compare your estimate to the market mid-price.
  3. If the gap (edge) exceeds MIN_EDGE, place a trade in the direction of the edge.

This file has two strategies:
  - MeanReversionStrategy  : fade extreme moves, bet on reversion to a prior
  - MomentumStrategy       : follow recent price trend

Extend these or add your own by subclassing BaseStrategy.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from kalshi_client import KalshiClient
import config

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    side: str          # "yes" or "no"
    action: str        # "buy" or "sell"
    contracts: int
    limit_price: int   # cents
    reason: str
    edge: float        # estimated edge in probability points


class BaseStrategy:
    def __init__(self, client: KalshiClient):
        self.client = client

    def analyze(self, market: dict) -> Optional[TradeSignal]:
        """Override this. Return a TradeSignal or None."""
        raise NotImplementedError

    def contracts_from_budget(self, price_cents: int, budget_usd: float) -> int:
        """How many contracts can we afford within budget?"""
        if price_cents <= 0:
            return 0
        cost_per = price_cents / 100  # dollars per contract
        return max(1, int(budget_usd / cost_per))


# ── Strategy 1: Mean Reversion ────────────────────────────────────────────────

class MeanReversionStrategy(BaseStrategy):
    """
    Uses a configurable prior probability for each market series.
    If market price deviates far from our prior, bet it will revert.

    Good for: events where you have external data (e.g. economist forecasts,
              historical base rates) giving you a prior belief.
    """

    # Map substring of ticker → your prior probability estimate (0.0–1.0).
    #
    # IMPORTANT: Only add a prior if you have a real external basis for it
    # (economist forecasts, historical base rates, etc.). A wrong prior
    # creates fake edge and will lose money. If unsure, leave it out and
    # let MomentumStrategy handle that market instead.
    #
    # KXFED tickers look like: KXFED-26MAR19-T4.25
    # Check the Kalshi site for exact threshold labels before adding priors.
    # KXFED tickers: KXFED-{MEETING_DATE}-T{RATE}
    # Each market asks "Will the upper bound of the fed funds rate be ABOVE T%
    # at the given FOMC meeting?"
    #
    # Set priors based on current rate + expected path.
    # Current upper bound as of Feb 2026: ~4.25%. Rates expected to drift down.
    # Verify/update these numbers before going live — check CME FedWatch tool.
    PRIORS = {
        "KXFED-T4.50": 0.10,  # above 4.50% — unlikely, would need a hike
        "KXFED-T4.25": 0.30,  # above 4.25% — possible if cuts are paused
        "KXFED-T4.00": 0.55,  # above 4.00% — roughly coin flip
        "KXFED-T3.75": 0.72,  # above 3.75% — more likely than not
        "KXFED-T3.50": 0.85,  # above 3.50% — likely
        "KXFED-T3.25": 0.92,  # above 3.25% — very likely
        "KXFED-T3.00": 0.96,  # above 3.00% — near certain
        # DO NOT add a blanket CPI prior — thresholds vary too much per release.
        # DO NOT add crypto/Nasdaq priors — let MomentumStrategy handle those.
    }

    def _get_prior(self, ticker: str) -> Optional[float]:
        for prefix, prob in self.PRIORS.items():
            if prefix in ticker.upper():
                return prob
        return None

    def analyze(self, market: dict) -> Optional[TradeSignal]:
        ticker = market["ticker"]
        prior = self._get_prior(ticker)
        if prior is None:
            logger.debug(f"No prior for {ticker}, skipping.")
            return None

        orderbook = self.client.get_orderbook(ticker)
        mid = self.client.mid_price(orderbook)
        if mid is None:
            logger.debug(f"No orderbook for {ticker}")
            return None

        market_prob = mid / 100  # convert cents → probability
        edge = prior - market_prob  # positive = market underprices YES

        if abs(edge) < config.MIN_EDGE:
            logger.debug(f"{ticker}: edge {edge:.3f} below threshold")
            return None

        if edge > 0:
            # Market underprices YES → buy YES
            side, action = "yes", "buy"
            limit_price = math.floor(mid)  # bid below ask
        else:
            # Market overprices YES → buy NO (equivalent to selling YES)
            side, action = "no", "buy"
            # NO price = 100 - YES price
            limit_price = math.floor(100 - mid)

        contracts = self.contracts_from_budget(limit_price, config.MAX_TRADE_USD)
        if contracts == 0:
            return None

        return TradeSignal(
            ticker=ticker,
            side=side,
            action=action,
            contracts=contracts,
            limit_price=limit_price,
            reason=f"mean_reversion | prior={prior:.2f} market={market_prob:.2f}",
            edge=abs(edge),
        )


# ── Strategy 2: Momentum ──────────────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Looks at recent trade history. If price has moved consistently in one
    direction over the last N trades, follow the trend.

    Good for: fast-moving markets (election nights, live event outcomes).
    Risky in slow markets — use only when you have high liquidity.
    """

    LOOKBACK = 10       # number of recent trades to examine
    MIN_MOVE = 5        # minimum net price move (cents) to trigger a signal

    def analyze(self, market: dict) -> Optional[TradeSignal]:
        ticker = market["ticker"]
        trades = self.client.get_market_history(ticker, limit=self.LOOKBACK)
        if len(trades) < 3:
            return None

        prices = [t["yes_price"] for t in trades if "yes_price" in t]
        if len(prices) < 3:
            return None

        net_move = prices[0] - prices[-1]  # newest - oldest

        if abs(net_move) < self.MIN_MOVE:
            return None

        orderbook = self.client.get_orderbook(ticker)
        mid = self.client.mid_price(orderbook)
        if mid is None:
            return None

        if net_move > 0:
            # Price moving up → buy YES
            side, action = "yes", "buy"
            limit_price = math.floor(mid)
        else:
            # Price moving down → buy NO
            side, action = "no", "buy"
            limit_price = math.floor(100 - mid)

        edge = abs(net_move) / 100

        contracts = self.contracts_from_budget(limit_price, config.MAX_TRADE_USD)
        if contracts == 0:
            return None

        return TradeSignal(
            ticker=ticker,
            side=side,
            action=action,
            contracts=contracts,
            limit_price=limit_price,
            reason=f"momentum | net_move={net_move:+}c over {len(prices)} trades",
            edge=edge,
        )


# ── Strategy 3: AI Sentiment (Gemini) ────────────────────────────────────────

class GeminiSentimentStrategy(BaseStrategy):
    """
    Uses Gemini to estimate the probability of a market resolving YES
    based on the market title/description.

    Requires GEMINI_API_KEY to be set in config.py.

    This is a qualitative overlay — best combined with a quantitative strategy.
    """

    # Max Gemini calls per bot cycle to stay within free-tier rate limits
    MAX_CALLS_PER_CYCLE = 5

    def __init__(self, client: KalshiClient):
        super().__init__(client)
        self._gemini = None
        self._calls_this_cycle = 0
        if config.GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini = genai.Client(api_key=config.GEMINI_API_KEY)
                logger.info("Gemini AI sentiment enabled.")
            except ImportError:
                logger.warning("google-genai not installed. Run: pip install google-genai")
        else:
            logger.info("GEMINI_API_KEY not set — AI sentiment disabled.")

    def _ask_gemini(self, question: str, context: str) -> Optional[float]:
        """
        Ask Gemini: what's the probability (0.0-1.0) that this resolves YES?
        Returns float or None on failure.
        """
        if not self._gemini:
            return None
        prompt = f"""
You are a prediction market analyst. Estimate the probability (0.0 to 1.0) that the following event resolves YES.
Respond with ONLY a decimal number between 0.0 and 1.0. No explanation.

Event: {question}
Additional context: {context}
"""
        try:
            response = self._gemini.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            text = response.text.strip()
            prob = float(text)
            return max(0.01, min(0.99, prob))
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                logger.warning("Gemini quota exhausted — disabling for this session.")
                self._gemini = None  # stop all further calls
            elif "ValueError" in type(e).__name__ or "AttributeError" in type(e).__name__:
                logger.warning(f"Gemini parse error: {e}")
            else:
                logger.warning(f"Gemini error: {e}")
            return None

    def analyze(self, market: dict) -> Optional[TradeSignal]:
        if not self._gemini:
            return None

        if self._calls_this_cycle >= self.MAX_CALLS_PER_CYCLE:
            return None

        ticker = market["ticker"]
        title = market.get("title", "")
        subtitle = market.get("subtitle", "")

        self._calls_this_cycle += 1
        prior = self._ask_gemini(title, subtitle)
        if prior is None:
            return None

        logger.info(f"Gemini estimate for {ticker}: {prior:.2f}")

        orderbook = self.client.get_orderbook(ticker)
        mid = self.client.mid_price(orderbook)
        if mid is None:
            return None

        market_prob = mid / 100
        edge = prior - market_prob

        if abs(edge) < config.MIN_EDGE:
            logger.debug(f"{ticker}: Gemini edge {edge:.3f} below threshold")
            return None

        if edge > 0:
            side, action = "yes", "buy"
            limit_price = math.floor(mid)
        else:
            side, action = "no", "buy"
            limit_price = math.floor(100 - mid)

        contracts = self.contracts_from_budget(limit_price, config.MAX_TRADE_USD)
        if contracts == 0:
            return None

        return TradeSignal(
            ticker=ticker,
            side=side,
            action=action,
            contracts=contracts,
            limit_price=limit_price,
            reason=f"gemini_sentiment | gemini={prior:.2f} market={market_prob:.2f}",
            edge=abs(edge),
        )


# ── Strategy 4: Crypto Price Model ────────────────────────────────────────────

class CryptoPriceStrategy(BaseStrategy):
    """
    Uses the real-time BTC/ETH spot price to estimate the probability that a
    daily contract resolves YES, then trades when the market price diverges.

    Model: assumes log-normal price, scales daily volatility by sqrt(hours_left/24)
    to get a fair-value probability for each threshold contract.

    Volatility is auto-calibrated from 30-day realized vol on startup using
    CoinGecko daily price history. Falls back to config.CRYPTO_DAILY_VOL if
    the fetch fails.

    Edge threshold is widened by half the bid-ask spread so that illiquid
    markets (wide spread = unreliable mid) require a stronger signal to trade.
    Markets with spread > CRYPTO_MAX_SPREAD_CENTS are skipped entirely.

    Set spot_prices before each cycle: strategy.spot_prices = client.get_spot_prices()
    """

    # Maps Kalshi series prefix → CoinGecko coin ID
    _COIN_IDS = {
        "KXBTCD": "bitcoin",
        "KXETH":  "ethereum",
    }

    def __init__(self, client: KalshiClient):
        super().__init__(client)
        self.spot_prices: dict = {}  # updated each cycle by bot.py
        self._realized_vols: dict = {}  # series → calibrated daily vol
        self._refresh_vols()

    def _refresh_vols(self):
        """
        Fetch 30-day daily price history from CoinGecko and compute realized
        daily volatility for each watched series. Falls back to config values
        if the fetch fails or there is insufficient data.
        """
        for series, coin_id in self._COIN_IDS.items():
            prices = self.client.get_price_history(coin_id, days=30)
            vol = self._compute_daily_vol(prices)
            if vol is not None:
                self._realized_vols[series] = vol
                logger.info(
                    f"Vol calibrated [{series}]: {vol:.4f} daily "
                    f"({vol * 100:.2f}%/day, {vol * math.sqrt(252) * 100:.1f}% annualized)"
                )
            else:
                fallback = config.CRYPTO_DAILY_VOL.get(series, 0.04)
                self._realized_vols[series] = fallback
                logger.warning(
                    f"Vol calibration failed [{series}] — using config fallback {fallback:.4f}"
                )

    @staticmethod
    def _compute_daily_vol(prices: list[float]) -> Optional[float]:
        """
        Compute realized daily volatility from daily closing prices.
        Returns sample std dev of log returns, or None if insufficient data.
        """
        if len(prices) < 5:
            return None
        log_returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if len(log_returns) < 4:
            return None
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF via math.erf."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _prob_above(self, spot: float, threshold: float, hours: float, daily_vol: float) -> float:
        """P(price > threshold at expiry) using log-normal model."""
        t = max(hours, 0.1) / 24.0          # fraction of a day
        vol = daily_vol * math.sqrt(t)
        if vol == 0:
            return 1.0 if spot > threshold else 0.0
        d = math.log(spot / threshold) / vol
        return max(0.01, min(0.99, self._norm_cdf(d)))

    def _hours_to_expiry(self, market: dict) -> float:
        t = market.get("close_time") or market.get("expiration_time")
        if not t:
            return 24.0
        close = datetime.fromisoformat(t.replace("Z", "+00:00"))
        return max(0.1, (close - datetime.now(timezone.utc)).total_seconds() / 3600)

    def _parse_ticker(self, ticker: str):
        """
        Returns (series, threshold, direction) from a ticker like:
          KXBTCD-26FEB2717-T76499.99  → ("KXBTCD", 76499.99, "above")
          KXETH-26FEB2118-B1960       → ("KXETH",  1960.0,   "below")
        Returns None on parse failure.
        """
        parts = ticker.split("-")
        if len(parts) < 3:
            return None
        series = parts[0]
        raw = parts[-1]           # e.g. "T76499.99" or "B1960"
        if not raw or raw[0] not in ("T", "B"):
            return None
        try:
            threshold = float(raw[1:])
        except ValueError:
            return None
        direction = "above" if raw[0] == "T" else "below"
        return series, threshold, direction

    def analyze(self, market: dict) -> Optional[TradeSignal]:
        ticker = market["ticker"]
        parsed = self._parse_ticker(ticker)
        if parsed is None:
            return None
        series, threshold, direction = parsed

        spot = self.spot_prices.get(series)
        if not spot:
            return None

        # Only trade near-the-money contracts
        if abs(spot - threshold) / spot > config.CRYPTO_NEAR_MONEY_PCT:
            return None

        daily_vol = self._realized_vols.get(series, config.CRYPTO_DAILY_VOL.get(series, 0.04))
        hours = self._hours_to_expiry(market)
        prob_above = self._prob_above(spot, threshold, hours, daily_vol)
        prior = prob_above if direction == "above" else (1.0 - prob_above)

        orderbook = self.client.get_orderbook(ticker)
        bid = self.client.best_yes_bid(orderbook)
        ask = self.client.best_yes_ask(orderbook)
        if bid is None or ask is None:
            return None

        # Skip illiquid markets with wide spreads — the mid is unreliable.
        spread = ask - bid
        if spread > config.CRYPTO_MAX_SPREAD_CENTS:
            logger.debug(
                f"SKIPPED {ticker}: spread={spread}c exceeds max {config.CRYPTO_MAX_SPREAD_CENTS}c"
            )
            return None

        mid = (bid + ask) / 2
        market_prob = mid / 100
        edge = prior - market_prob

        # Widen the edge threshold by half the spread in probability terms.
        # A 3-cent spread requires MIN_EDGE + 0.015 rather than plain MIN_EDGE,
        # because a wider spread means the mid is a less precise signal.
        adjusted_min_edge = config.MIN_EDGE + (spread / 200)
        if abs(edge) < adjusted_min_edge:
            return None

        # If model and market disagree by too much, the market likely has
        # information we don't (momentum, news, order flow). Skip the trade.
        if abs(edge) > config.MAX_MODEL_MARKET_DISAGREEMENT:
            logger.info(
                f"SKIPPED {ticker}: model={prior:.2f} market={market_prob:.2f} "
                f"disagreement={abs(edge):.2f} exceeds cap of {config.MAX_MODEL_MARKET_DISAGREEMENT}"
            )
            return None

        if edge > 0:
            side, action = "yes", "buy"
            limit_price = math.floor(mid)
        else:
            side, action = "no", "buy"
            limit_price = math.floor(100 - mid)

        contracts = self.contracts_from_budget(limit_price, config.MAX_TRADE_USD)
        if contracts == 0:
            return None

        return TradeSignal(
            ticker=ticker,
            side=side,
            action=action,
            contracts=contracts,
            limit_price=limit_price,
            reason=(
                f"crypto_price | spot={spot:.0f} threshold={threshold:.0f} ({direction}) "
                f"prior={prior:.2f} market={market_prob:.2f} "
                f"vol={daily_vol:.4f} spread={spread}c hours_left={hours:.1f}"
            ),
            edge=abs(edge),
        )
