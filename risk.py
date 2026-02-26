"""
Risk management layer.
Sits between the strategy and order execution — gates all trades.
"""

import logging
from datetime import datetime, timezone
from kalshi_client import KalshiClient
from strategy import TradeSignal
import config

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, client: KalshiClient):
        self.client = client
        self._realized_pnl_today = 0.0

    def refresh_realized_pnl(self):
        """
        Recompute today's realized P&L from Kalshi fills and settlements.
        Call this once at the start of each bot cycle (not per-signal) to
        keep the daily loss limit accurate without hammering the API.

        Logic:
          - Settlements tell us what we got paid for closed positions today.
          - Fills tell us what we paid to open those positions today.
          - Realized P&L = settlement revenue - fill costs, for settled tickers.

        Only settled tickers are counted — open positions are covered by
        _estimate_unrealized() in check(), so there's no double-counting.
        """
        today_midnight = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Kalshi uses Unix timestamps in seconds for min_ts on fills/settlements.
        min_ts_s = int(today_midnight.timestamp())

        try:
            settlements = self.client.get_settlements(min_ts=min_ts_s)
            fills = self.client.get_fills(min_ts=min_ts_s)
        except Exception as e:
            logger.warning(f"Could not fetch fills/settlements for P&L: {e}")
            return

        if not settlements:
            self._realized_pnl_today = 0.0
            return

        # Build cost basis per ticker from today's buy fills.
        # Each fill: count contracts at a given yes_price.
        # For YES buys the cost is yes_price per contract.
        # For NO buys the cost is (100 - yes_price) per contract.
        cost_by_ticker: dict[str, float] = {}
        for fill in fills:
            if fill.get("action") != "buy":
                continue
            ticker = fill.get("ticker", "")
            count = fill.get("count", 0)
            yes_price = fill.get("yes_price", 0)
            side = fill.get("side", "yes")
            price_per = yes_price if side == "yes" else (100 - yes_price)
            cost_by_ticker[ticker] = cost_by_ticker.get(ticker, 0.0) + (count * price_per) / 100

        # For each settled position, compute revenue minus cost basis.
        realized = 0.0
        for s in settlements:
            ticker = s.get("ticker", "")
            revenue = s.get("revenue", 0) / 100  # cents → dollars
            cost = cost_by_ticker.get(ticker, 0.0)
            pnl = revenue - cost
            logger.debug(f"Settlement {ticker}: revenue=${revenue:.2f} cost=${cost:.2f} pnl=${pnl:.2f}")
            realized += pnl

        self._realized_pnl_today = realized
        logger.info(f"Today's realized P&L: ${realized:.2f} ({len(settlements)} settlements)")

    def check(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Call this before every order.
        refresh_realized_pnl() should be called once per cycle before any check().
        """
        # 1. Daily loss limit — combines settled losses with current unrealized
        balance = self.client.get_balance()
        positions = self.client.get_portfolio()
        unrealized = self._estimate_unrealized(positions)
        total_loss = min(0, unrealized + self._realized_pnl_today)

        if abs(total_loss) >= config.DAILY_LOSS_LIMIT_USD:
            return False, (
                f"Daily loss limit hit (realized=${self._realized_pnl_today:.2f} "
                f"unrealized=${unrealized:.2f} total=${abs(total_loss):.2f} "
                f">= ${config.DAILY_LOSS_LIMIT_USD})"
            )

        # 2. Position count
        open_orders = self.client.get_open_orders()
        if len(open_orders) >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached ({len(open_orders)})"

        # 3. Trade size sanity
        cost = (signal.limit_price / 100) * signal.contracts
        if cost > config.MAX_TRADE_USD:
            return False, f"Trade cost ${cost:.2f} exceeds MAX_TRADE_USD ${config.MAX_TRADE_USD}"

        # 4. Sufficient balance
        if cost > balance:
            return False, f"Insufficient balance (${balance:.2f}) for ${cost:.2f} trade"

        # 5. Minimum edge
        if signal.edge < config.MIN_EDGE:
            return False, f"Edge {signal.edge:.3f} below minimum {config.MIN_EDGE}"

        return True, "approved"

    def _estimate_unrealized(self, positions: list[dict]) -> float:
        """
        Rough unrealized P&L estimate using realized_pnl from open positions.
        Kalshi does not return current mark-to-market value in the portfolio
        response, so we use realized_pnl (partial closes, adjustments) as a
        conservative proxy. The stop-loss mechanism handles cutting losing open
        positions; settled losses are captured by refresh_realized_pnl().
        """
        total = 0.0
        for pos in positions:
            if pos.get("position", 0) == 0:
                continue  # skip closed/historical positions
            total += pos.get("realized_pnl", 0) / 100  # cents → dollars
        return total
