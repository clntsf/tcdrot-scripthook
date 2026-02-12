# RIT Client REST API wrapper
# Implements the full RIT REST API v1.0.3 specification
# https://rit.306w.ca/RIT-REST-API/1.0.3/

import requests


class TradingAPI:

    def __init__(self, api_key, host="localhost", port=9999):
        self.base_url = f"http://{host}:{port}/v1"
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def _get(self, path, params=None):
        resp = self.session.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path, params=None):
        resp = self.session.post(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path, params=None):
        resp = self.session.delete(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Case ──

    def get_case(self):
        """Gets information about the current case."""
        return self._get("/case")

    # ── Trader ──

    def get_trader(self):
        """Gets information about the currently signed in trader."""
        return self._get("/trader")

    # ── Limits ──

    def get_limits(self):
        """Gets trading limits for the current case."""
        return self._get("/limits")

    # ── News ──

    def get_news(self, since=None, limit=None):
        """Gets the most recent news."""
        params = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        return self._get("/news", params or None)

    # ── Assets ──

    def get_assets(self, ticker=None):
        """Gets a list of available assets."""
        params = {}
        if ticker is not None:
            params["ticker"] = ticker
        return self._get("/assets", params or None)

    def get_assets_history(self, ticker=None, period=None, limit=None):
        """Gets the activity log for assets."""
        params = {}
        if ticker is not None:
            params["ticker"] = ticker
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self._get("/assets/history", params or None)

    # ── Securities ──

    def get_securities(self, ticker=None):
        """Gets a list of available securities and associated positions."""
        params = {}
        if ticker is not None:
            params["ticker"] = ticker
        return self._get("/securities", params or None)

    def get_securities_book(self, ticker, limit=None):
        """Gets the order book of a security."""
        params = {"ticker": ticker}
        if limit is not None:
            params["limit"] = limit
        return self._get("/securities/book", params)

    def get_securities_history(self, ticker, period=None, limit=None):
        """Gets the OHLC history for a security."""
        params = {"ticker": ticker}
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self._get("/securities/history", params)

    def get_securities_tas(self, ticker, after=None, period=None, limit=None):
        """Gets time & sales history for a security."""
        params = {"ticker": ticker}
        if after is not None:
            params["after"] = after
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self._get("/securities/tas", params)

    # ── Orders ──

    def get_orders(self, status=None):
        """Gets a list of all orders. Defaults to OPEN."""
        params = {}
        if status is not None:
            params["status"] = status
        return self._get("/orders", params or None)

    def post_order(self, ticker, type, quantity, action, price=None, dry_run=None):
        """Insert a new order. Rate-limited."""
        params = {
            "ticker": ticker,
            "type": type,
            "quantity": quantity,
            "action": action,
        }
        if price is not None:
            params["price"] = price
        if dry_run is not None:
            params["dry_run"] = dry_run
        return self._post("/orders", params)

    def get_order(self, order_id):
        """Gets the details of a specific order."""
        return self._get(f"/orders/{order_id}")

    def cancel_order(self, order_id):
        """Cancel an open order."""
        return self._delete(f"/orders/{order_id}")

    # ── Tenders ──

    def get_tenders(self):
        """Gets a list of all active tenders."""
        return self._get("/tenders")

    def accept_tender(self, tender_id, price=None):
        """Accept a tender. Price required if not fixed-bid."""
        params = {}
        if price is not None:
            params["price"] = price
        return self._post(f"/tenders/{tender_id}", params or None)

    def decline_tender(self, tender_id):
        """Decline a tender."""
        return self._delete(f"/tenders/{tender_id}")

    # ── Leases ──

    def get_leases(self):
        """List of all assets currently being leased or used."""
        return self._get("/leases")

    def post_lease(self, ticker, from1=None, quantity1=None,
                   from2=None, quantity2=None, from3=None, quantity3=None):
        """Lease or use an asset."""
        params = {"ticker": ticker}
        if from1 is not None:
            params["from1"] = from1
        if quantity1 is not None:
            params["quantity1"] = quantity1
        if from2 is not None:
            params["from2"] = from2
        if quantity2 is not None:
            params["quantity2"] = quantity2
        if from3 is not None:
            params["from3"] = from3
        if quantity3 is not None:
            params["quantity3"] = quantity3
        return self._post("/leases", params)

    def get_lease(self, lease_id):
        """Gets the details of a specific lease."""
        return self._get(f"/leases/{lease_id}")

    def use_lease(self, lease_id, from1, quantity1,
                  from2=None, quantity2=None, from3=None, quantity3=None):
        """Use a leased asset."""
        params = {"from1": from1, "quantity1": quantity1}
        if from2 is not None:
            params["from2"] = from2
        if quantity2 is not None:
            params["quantity2"] = quantity2
        if from3 is not None:
            params["from3"] = from3
        if quantity3 is not None:
            params["quantity3"] = quantity3
        return self._post(f"/leases/{lease_id}", params)

    def unlease(self, lease_id):
        """Unlease an asset."""
        return self._delete(f"/leases/{lease_id}")

    # ── Commands ──

    def cancel_orders(self, all=None, ticker=None, ids=None, query=None):
        """Bulk cancel open orders. Specify exactly one parameter."""
        params = {}
        if all is not None:
            params["all"] = all
        if ticker is not None:
            params["ticker"] = ticker
        if ids is not None:
            params["ids"] = ids
        if query is not None:
            params["query"] = query
        return self._post("/commands/cancel", params or None)

    # ── Convenience methods (backward compatible) ──

    def buy(self, ticker, quantity, price=None):
        """Submit a buy order. LIMIT if price given, MARKET otherwise."""
        if price is not None:
            return self.post_order(ticker, "LIMIT", quantity, "BUY", price=price)
        return self.post_order(ticker, "MARKET", quantity, "BUY")

    def sell(self, ticker, quantity, price=None):
        """Submit a sell order. LIMIT if price given, MARKET otherwise."""
        if price is not None:
            return self.post_order(ticker, "LIMIT", quantity, "SELL", price=price)
        return self.post_order(ticker, "MARKET", quantity, "SELL")
