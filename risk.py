"""
Risk management layer.
Sits between the strategy and order execution — gates all trades.
"""

import logging
from kalshi_client import KalshiClient
from strategy import TradeSignal
import config

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, client: KalshiClient):
        self.client = client
        self._realized_pnl_today = 0.0  # updated as trades close

    def check(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Call this before every order.
        """
        # 1. Daily loss limit
        balance = self.client.get_balance()
        positions = self.client.get_portfolio()
        unrealized = self._estimate_unrealized(positions)
        total_loss = min(0, unrealized + self._realized_pnl_today)

        if abs(total_loss) >= config.DAILY_LOSS_LIMIT_USD:
            return False, f"Daily loss limit hit (${abs(total_loss):.2f} >= ${config.DAILY_LOSS_LIMIT_USD})"

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
        Rough unrealized P&L estimate.
        Kalshi positions include market_exposure and current value.
        """
        total = 0.0
        for pos in positions:
            # resting_orders_count > 0 means open
            value = pos.get("value", 0) / 100       # cents → dollars
            cost = pos.get("total_cost", 0) / 100
            total += value - cost
        return total
