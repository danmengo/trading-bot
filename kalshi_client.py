"""
Kalshi REST API client with RSA-PSS request signing.
Docs: https://docs.kalshi.com/getting_started/api_keys
"""

import base64
import datetime
import logging
import uuid
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

logger = logging.getLogger(__name__)


def _load_private_key(pem_text: str):
    """Load an RSA private key from a PEM string."""
    return serialization.load_pem_private_key(pem_text.encode(), password=None)


def _sign_pss(private_key, message: str) -> str:
    """RSA-PSS sign a string and return base64-encoded signature."""
    sig = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


class KalshiClient:
    def __init__(self):
        self.base_url = config.KALSHI_BASE_URLS[config.KALSHI_ENV]
        self.key_id = config.KALSHI_KEY_ID
        self.private_key = _load_private_key(config.KALSHI_PRIVATE_KEY)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _auth_headers(self, method: str, path: str) -> dict:
        """Build the three required Kalshi auth headers for this request."""
        ts_ms = str(int(datetime.datetime.now().timestamp() * 1000))
        # Kalshi requires signing the FULL path including /trade-api/v2 prefix,
        # with query string stripped.
        full_path = ("/trade-api/v2" + path).split("?")[0]
        msg = ts_ms + method.upper() + full_path
        signature = _sign_pss(self.private_key, msg)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, headers=self._auth_headers("GET", path))
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, json=body, headers=self._auth_headers("POST", path))
        if not resp.ok:
            logger.error(f"POST {path} {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.delete(url, headers=self._auth_headers("DELETE", path))
        resp.raise_for_status()
        return resp.json()

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Returns available balance in USD."""
        data = self._get("/portfolio/balance")
        return data.get("balance", 0) / 100

    def get_portfolio(self) -> list[dict]:
        """Returns current open positions."""
        data = self._get("/portfolio/positions")
        return data.get("market_positions", [])

    # ── Markets ───────────────────────────────────────────────────────────────

    def get_markets(self, series_ticker: str = None, limit: int = 100) -> list[dict]:
        """List markets, optionally filtered by series prefix."""
        params = {"limit": limit, "status": "open"}
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = self._get("/markets", params=params)
        return data.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        data = self._get(f"/markets/{ticker}")
        return data.get("market", {})

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """Returns orderbook for a market."""
        data = self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})
        return data.get("orderbook", {})

    def get_market_history(self, ticker: str, limit: int = 100) -> list[dict]:
        """Recent trade history for a market."""
        data = self._get("/markets/trades", params={"ticker": ticker, "limit": limit})
        return data.get("trades", [])

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        price_cents: int,
        order_type: str = "limit",
    ) -> dict:
        yes_price = price_cents if side == "yes" else (100 - price_cents)
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": action,
            "side": side,
            "count": contracts,
            "type": order_type,
            "yes_price": yes_price,
        }
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] Would place order: {body}")
            return {"status": "dry_run", "order": body}
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        if config.DRY_RUN:
            logger.info(f"[DRY RUN] Would cancel order: {order_id}")
            return {"status": "dry_run"}
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_open_orders(self) -> list[dict]:
        data = self._get("/portfolio/orders", params={"status": "resting"})
        return data.get("orders", [])

    # ── Spot prices ───────────────────────────────────────────────────────────

    def get_spot_prices(self) -> dict:
        """Fetch BTC and ETH spot prices from CoinGecko (free, no API key needed).
        Returns e.g. {"KXBTCD": 68200.0, "KXETH": 2050.0}"""
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "KXBTCD": data.get("bitcoin", {}).get("usd"),
                "KXETH":  data.get("ethereum", {}).get("usd"),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch spot prices: {e}")
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────────
    # Kalshi orderbook structure:
    #   {"yes": [[price, size], ...], "no": [[price, size], ...]}
    #   "yes" = bids for YES (buyers), sorted descending by price
    #   "no"  = bids for NO  (buyers), sorted descending by price
    #   YES ask = 100 - best NO bid  (since YES + NO = 100 cents)

    def best_yes_bid(self, orderbook: dict) -> Optional[int]:
        """Highest price (cents) a buyer will pay for YES."""
        yes_bids = orderbook.get("yes", [])
        return yes_bids[0][0] if yes_bids else None

    def best_yes_ask(self, orderbook: dict) -> Optional[int]:
        """Lowest price (cents) a seller will accept for YES (implied from NO bids)."""
        no_bids = orderbook.get("no", [])
        return (100 - no_bids[0][0]) if no_bids else None

    def mid_price(self, orderbook: dict) -> Optional[float]:
        bid = self.best_yes_bid(orderbook)
        ask = self.best_yes_ask(orderbook)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
