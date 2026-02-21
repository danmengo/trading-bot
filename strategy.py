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

    # Map series prefix → your prior probability estimate (0.0–1.0)
    # Tune these based on external data / research.
    PRIORS = {
        "FED-PAUSE":   0.70,   # Fed likely to hold rates
        "FED-HIKE":    0.10,
        "FED-CUT":     0.20,
        "CPI-":        0.50,   # neutral for generic CPI markets
        "INXD-":       0.50,   # S&P daily move: roughly 50/50
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
