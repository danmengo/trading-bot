"""
Main bot loop.

Runs strategies against watched markets, applies risk checks,
and submits orders (or logs them in DRY_RUN mode).
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from kalshi_client import KalshiClient
from strategy import GeminiSentimentStrategy, CryptoPriceStrategy
from risk import RiskManager
import config
import trade_log

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

        self.crypto_price_strategy = CryptoPriceStrategy(self.client)

        # Add or remove strategies here
        self.strategies = [
            self.crypto_price_strategy,              # spot-price probability model
            GeminiSentimentStrategy(self.client),    # no-ops if GEMINI_API_KEY is empty
        ]

    def run_once(self):
        """Single scan of all watched markets."""
        # Reset Gemini call counter for this cycle
        for s in self.strategies:
            if hasattr(s, "_calls_this_cycle"):
                s._calls_this_cycle = 0

        # Refresh realized P&L once per cycle so the daily loss limit is accurate
        # even after positions settle and disappear from the portfolio.
        self.risk.refresh_realized_pnl()

        balance = self.client.get_balance()
        logger.info(f"Balance: ${balance:.2f} | DRY_RUN={config.DRY_RUN}")

        # Fetch spot prices once per cycle and share with price strategy
        spot_prices = self.client.get_spot_prices()
        if spot_prices:
            logger.info(f"Spot prices: BTC=${spot_prices.get('KXBTCD', '?'):,.0f}  ETH=${spot_prices.get('KXETH', '?'):,.0f}")
        self.crypto_price_strategy.spot_prices = spot_prices

        # Cancel stale resting orders and build busy_series from both resting orders
        # AND filled portfolio positions — prevents stacking on the same underlying.
        try:
            open_orders = self.client.get_open_orders()
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=config.ORDER_CANCEL_AFTER_MINUTES)
            remaining_orders = []
            for order in open_orders:
                created = order.get("created_time")
                order_id = order.get("order_id")
                if created and order_id:
                    order_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if order_time < cutoff:
                        logger.info(
                            f"Cancelling stale order {order_id} ({order.get('ticker')}) "
                            f"— unfilled after {config.ORDER_CANCEL_AFTER_MINUTES}m"
                        )
                        try:
                            self.client.cancel_order(order_id)
                        except Exception as e:
                            logger.warning(f"Failed to cancel order {order_id}: {e}")
                            remaining_orders.append(order)
                        continue
                remaining_orders.append(order)

            # Fetch positions once — used for both stop-loss checks and busy_series.
            try:
                positions = self.client.get_portfolio()
            except Exception as e:
                logger.warning(f"Could not fetch portfolio positions: {e}")
                positions = []

            # Check stop-losses before scanning for new trades.
            self._check_stops(positions, remaining_orders)

            # Series with resting orders
            busy_series = {
                ticker.split("-")[0]
                for order in remaining_orders
                for ticker in [order.get("ticker", "")]
                if ticker
            }

            # Also block series where we already hold a filled position.
            for pos in positions:
                ticker = pos.get("ticker", "")
                if ticker:
                    busy_series.add(ticker.split("-")[0])

            if busy_series:
                logger.info(f"Skipping series with existing orders/positions: {busy_series}")
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")
            busy_series = set()

        for series in config.WATCHED_SERIES:
            if series in busy_series:
                logger.info(f"Skipping {series} — resting order already open")
                continue

            try:
                markets = self.client.get_markets(series_ticker=series)
            except Exception as e:
                logger.error(f"Failed to fetch markets for {series}: {e}")
                continue

            # Filter by time to expiry — only trade within the configured window.
            # Too close: model breaks down. Too far: probabilities compress toward 50%.
            now = datetime.now(timezone.utc)
            def hours_to_expiry(m):
                t = m.get("close_time") or m.get("expiration_time")
                if not t:
                    return float("inf")
                return (datetime.fromisoformat(t.replace("Z", "+00:00")) - now).total_seconds() / 3600
            markets = [
                m for m in markets
                if config.MIN_HOURS_TO_EXPIRY <= hours_to_expiry(m) <= config.MAX_HOURS_TO_EXPIRY
            ]

            logger.info(
                f"Found {len(markets)} open markets in series '{series}' "
                f"(window: {config.MIN_HOURS_TO_EXPIRY}–{config.MAX_HOURS_TO_EXPIRY}h to expiry)"
            )

            # Find the best signal across all markets in this series and execute it.
            # One trade per series per cycle prevents correlated bets on the same underlying.
            best_signal = None
            best_strategy_name = None

            for market in markets:
                ticker = market["ticker"]
                for strategy in self.strategies:
                    try:
                        signal = strategy.analyze(market)
                    except Exception as e:
                        logger.warning(f"{strategy.__class__.__name__} error on {ticker}: {e}")
                        continue
                    if signal and (best_signal is None or signal.edge > best_signal.edge):
                        best_signal = signal
                        best_strategy_name = strategy.__class__.__name__

            if best_signal is None:
                continue

            approved, reason = self.risk.check(best_signal)
            if not approved:
                logger.info(f"BLOCKED {best_signal.ticker}: {reason}")
                continue

            cost = (best_signal.limit_price / 100) * best_signal.contracts
            logger.info(
                f"SIGNAL [{best_strategy_name}] {best_signal.ticker} | "
                f"{best_signal.action.upper()} {best_signal.contracts}x {best_signal.side.upper()} "
                f"@ {best_signal.limit_price}c (${cost:.2f}) | edge={best_signal.edge:.3f} | "
                f"{best_signal.reason}"
            )
            try:
                result = self.client.place_order(
                    ticker=best_signal.ticker,
                    side=best_signal.side,
                    action=best_signal.action,
                    contracts=best_signal.contracts,
                    price_cents=best_signal.limit_price,
                )
                logger.info(f"Order result: {result}")
                trade_log.log_trade(best_signal, best_strategy_name, result, config.DRY_RUN)
            except Exception as e:
                logger.error(f"Order failed for {best_signal.ticker}: {e}")

        # ── Specific tickers ──────────────────────────────────────────────────
        for ticker in config.WATCHED_TICKERS:
            try:
                market = self.client.get_market(ticker)
                if not market:
                    logger.warning(f"Ticker {ticker} not found or not open")
                    continue
                logger.info(f"Scanning specific ticker: {ticker}")
                self._trade_market(market)
            except Exception as e:
                logger.error(f"Failed to fetch ticker {ticker}: {e}")

    def _check_stops(self, positions: list, open_orders: list):
        """
        Check all open positions for stop-loss conditions each cycle.
        Exits at the best available bid for our side — never at mid, since
        on Kalshi you can only sell at what buyers are willing to pay.

        Kalshi field mapping (from debug_positions.py):
          position        → number of contracts held (0 = closed/historical, skip)
          market_exposure → cost basis of the current position in cents
          Current value is not provided — fetched live from the orderbook.

        Skips any position that already has a pending sell order to avoid
        stacking duplicate exits across cycles.
        """
        # Don't re-submit a sell if one is already resting for this ticker.
        pending_sells = {
            order.get("ticker")
            for order in open_orders
            if order.get("action") == "sell"
        }

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker or ticker in pending_sells:
                continue

            # position = 0 means the position is closed or historical — skip it.
            count = pos.get("position", 0)
            if count == 0:
                continue

            # market_exposure = cost basis in cents for the contracts currently held.
            cost_cents = pos.get("market_exposure", 0)
            if cost_cents <= 0:
                continue

            # Positive position = YES contracts. Negative would be NO.
            side = "yes" if count > 0 else "no"
            count = abs(count)

            # Fetch orderbook to get both the current bid (exit price) and
            # current market value — Kalshi doesn't provide mark-to-market in portfolio.
            orderbook = self.client.get_orderbook(ticker)
            if side == "yes":
                bid = self.client.best_yes_bid(orderbook)
            else:
                bid = self.client.best_no_bid(orderbook)

            if not bid or bid <= 0:
                logger.warning(
                    f"STOP-LOSS: no {side} bid for {ticker} — cannot exit, will retry next cycle"
                )
                continue

            # Current value = what we'd receive if we sold at the best bid right now.
            current_value_cents = bid * count
            loss_pct = (cost_cents - current_value_cents) / cost_cents

            if loss_pct < config.STOP_LOSS_PCT:
                continue

            logger.info(
                f"STOP-LOSS triggered {ticker} | {side.upper()} {count}x | "
                f"paid=${cost_cents / 100:.2f} now=${current_value_cents / 100:.2f} "
                f"loss={loss_pct:.0%} → selling at {bid}c"
            )
            try:
                result = self.client.place_order(
                    ticker=ticker,
                    side=side,
                    action="sell",
                    contracts=count,
                    price_cents=bid,
                )
                logger.info(f"Stop-loss order result: {result}")
            except Exception as e:
                logger.error(f"Stop-loss order failed for {ticker}: {e}")

    def _trade_market(self, market: dict):
        """Evaluate all strategies on a single market and execute the best signal."""
        ticker = market["ticker"]
        best_signal = None
        best_strategy_name = None

        for strategy in self.strategies:
            try:
                signal = strategy.analyze(market)
            except Exception as e:
                logger.warning(f"{strategy.__class__.__name__} error on {ticker}: {e}")
                continue
            if signal and (best_signal is None or signal.edge > best_signal.edge):
                best_signal = signal
                best_strategy_name = strategy.__class__.__name__

        if best_signal is None:
            logger.info(f"No signal for {ticker}")
            return

        approved, reason = self.risk.check(best_signal)
        if not approved:
            logger.info(f"BLOCKED {ticker}: {reason}")
            return

        cost = (best_signal.limit_price / 100) * best_signal.contracts
        logger.info(
            f"SIGNAL [{best_strategy_name}] {ticker} | "
            f"{best_signal.action.upper()} {best_signal.contracts}x {best_signal.side.upper()} "
            f"@ {best_signal.limit_price}c (${cost:.2f}) | edge={best_signal.edge:.3f} | "
            f"{best_signal.reason}"
        )
        try:
            result = self.client.place_order(
                ticker=best_signal.ticker,
                side=best_signal.side,
                action=best_signal.action,
                contracts=best_signal.contracts,
                price_cents=best_signal.limit_price,
            )
            logger.info(f"Order result: {result}")
            trade_log.log_trade(best_signal, best_strategy_name, result, config.DRY_RUN)
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
