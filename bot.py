"""
Main bot loop.

Runs strategies against watched markets, applies risk checks,
and submits orders (or logs them in DRY_RUN mode).
"""

import logging
import time
from kalshi_client import KalshiClient
from strategy import MeanReversionStrategy, MomentumStrategy, GeminiSentimentStrategy
from risk import RiskManager
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bot")


class TradingBot:
    def __init__(self):
        self.client = KalshiClient()
        self.risk = RiskManager(self.client)

        # Add or remove strategies here
        self.strategies = [
            MeanReversionStrategy(self.client),
            MomentumStrategy(self.client),
            GeminiSentimentStrategy(self.client),   # no-ops if GEMINI_API_KEY is empty
        ]

    def run_once(self):
        """Single scan of all watched markets."""
        # Reset Gemini call counter for this cycle
        for s in self.strategies:
            if hasattr(s, "_calls_this_cycle"):
                s._calls_this_cycle = 0

        balance = self.client.get_balance()
        logger.info(f"Balance: ${balance:.2f} | DRY_RUN={config.DRY_RUN}")

        for series in config.WATCHED_SERIES:
            try:
                markets = self.client.get_markets(series_ticker=series)
            except Exception as e:
                logger.error(f"Failed to fetch markets for {series}: {e}")
                continue

            logger.info(f"Found {len(markets)} open markets in series '{series}'")

            for market in markets:
                ticker = market["ticker"]

                for strategy in self.strategies:
                    try:
                        signal = strategy.analyze(market)
                    except Exception as e:
                        logger.warning(f"{strategy.__class__.__name__} error on {ticker}: {e}")
                        continue

                    if signal is None:
                        continue

                    # Risk gate
                    approved, reason = self.risk.check(signal)
                    if not approved:
                        logger.info(f"BLOCKED {ticker}: {reason}")
                        continue

                    # Execute
                    cost = (signal.limit_price / 100) * signal.contracts
                    logger.info(
                        f"SIGNAL [{strategy.__class__.__name__}] {ticker} | "
                        f"{signal.action.upper()} {signal.contracts}x {signal.side.upper()} "
                        f"@ {signal.limit_price}c (${cost:.2f}) | edge={signal.edge:.3f} | "
                        f"{signal.reason}"
                    )

                    try:
                        result = self.client.place_order(
                            ticker=signal.ticker,
                            side=signal.side,
                            action=signal.action,
                            contracts=signal.contracts,
                            price_cents=signal.limit_price,
                        )
                        logger.info(f"Order result: {result}")
                    except Exception as e:
                        logger.error(f"Order failed for {ticker}: {e}")

    def run(self):
        """Main loop — polls forever."""
        logger.info("=" * 60)
        logger.info("Kalshi Trading Bot starting")
        logger.info(f"  Environment : {config.KALSHI_ENV}")
        logger.info(f"  DRY_RUN     : {config.DRY_RUN}")
        logger.info(f"  Strategies  : {[s.__class__.__name__ for s in self.strategies]}")
        logger.info(f"  Poll every  : {config.POLL_INTERVAL_SECONDS}s")
        logger.info("=" * 60)

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Interrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

            logger.info(f"Sleeping {config.POLL_INTERVAL_SECONDS}s...")
            time.sleep(config.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
