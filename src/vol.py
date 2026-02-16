"""
Volatility Arbitrage Trading Module for RITC 2026
By Claude

This module implements volatility arbitrage strategies by:
1. Calculating implied volatility from market prices
2. Comparing to analyst-forecasted realized volatility
3. Identifying mispriced options
4. Executing trades with delta hedging
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from api import TradingAPI

# Option ticker constants
STRIKES = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
CALL_TICKERS = [f"RTM1C{strike}" for strike in STRIKES]
PUT_TICKERS = [f"RTM1P{strike}" for strike in STRIKES]
ALL_OPTION_TICKERS = CALL_TICKERS + PUT_TICKERS

API_KEY = "YOUR_API_KEY"

# Trading constants
RISK_FREE_RATE = 0.0  # r = 0% as specified
TRADING_YEAR_DAYS = 240  # 240 trading days per year
MONTH_DAYS = 20  # 20 trading days per month
OPTION_MULTIPLIER = 100  # Each contract represents 100 shares


@dataclass
class OptionPosition:
    """Represents a position in an option"""
    ticker: str
    quantity: int  # Number of contracts (positive = long, negative = short)
    strike: float
    is_call: bool
    delta: float
    entry_price: float = 0.0


@dataclass
class TradeSignal:
    """Represents a trading signal for an option"""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    market_price: float
    fair_value: float
    edge: float
    implied_vol: float
    realized_vol: float
    delta: float


class BlackScholesCalculator:
    """Black-Scholes option pricing calculator"""
    
    @staticmethod
    def norm_cdf(x: float) -> float:
        """
        Cumulative distribution function for standard normal distribution.
        Uses polynomial approximation for accuracy without external dependencies.
        """
        # Constants for approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2.0)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility (annualized)
        
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0 or sigma <= 0:
            # Handle edge cases
            if T <= 0:
                return (float('inf') if S > K else float('-inf')), (float('inf') if S > K else float('-inf'))
            if sigma <= 0:
                sigma = 0.0001  # Small non-zero value
        
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2
    
    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price using Black-Scholes"""
        if T <= 0:
            return max(0, S - K)
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        
        call = S * cls.norm_cdf(d1) - K * math.exp(-r * T) * cls.norm_cdf(d2)
        return max(0, call)
    
    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price using Black-Scholes"""
        if T <= 0:
            return max(0, K - S)
        
        d1, d2 = cls.calculate_d1_d2(S, K, T, r, sigma)
        
        put = K * math.exp(-r * T) * cls.norm_cdf(-d2) - S * cls.norm_cdf(-d1)
        return max(0, put)
    
    @classmethod
    def call_delta(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate delta for call option"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma)
        return cls.norm_cdf(d1)
    
    @classmethod
    def put_delta(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate delta for put option"""
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1, _ = cls.calculate_d1_d2(S, K, T, r, sigma)
        return cls.norm_cdf(d1) - 1.0
    
    @classmethod
    def implied_volatility(cls, market_price: float, S: float, K: float, T: float, 
                          r: float, is_call: bool, tol: float = 0.0001, max_iter: int = 100) -> Optional[float]:
        """
        Calculate implied volatility using bisection method.
        
        Args:
            market_price: Observed market price
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            is_call: True for call, False for put
            tol: Tolerance for convergence
            max_iter: Maximum iterations
        
        Returns:
            Implied volatility or None if not found
        """
        if T <= 0:
            return None
        
        # Check if price is arbitrage-free
        intrinsic = max(0, S - K) if is_call else max(0, K - S)
        if market_price < intrinsic - tol:
            return None  # Price below intrinsic value
        
        # Bisection method bounds
        sigma_low = 0.001  # 0.1% vol
        sigma_high = 5.0   # 500% vol
        
        price_func = cls.call_price if is_call else cls.put_price
        
        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid = price_func(S, K, T, r, sigma_mid)
            
            if abs(price_mid - market_price) < tol:
                return sigma_mid
            
            if price_mid < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
        
        # Return best estimate if didn't converge
        return (sigma_low + sigma_high) / 2


class VolatilityArbitrageTrader:
    """Main trading logic for volatility arbitrage"""
    
    def __init__(self, rtm_price: float, time_to_expiration_days: float = 20):
        """
        Initialize the trader.
        
        Args:
            rtm_price: Current price of RTM ETF
            time_to_expiration_days: Days until option expiration (default 20)
        """
        self.rtm_price = rtm_price
        self.time_to_expiration = time_to_expiration_days / TRADING_YEAR_DAYS
        self.bs_calc = BlackScholesCalculator()
        self.positions: List[OptionPosition] = []
        self.rtm_position = 0  # Net RTM shares held
        self.last_realized_vol = 0.35  # Updated each loop tick from analyst feed
        self.tapi = TradingAPI(API_KEY)
    
    def update_time_to_expiration(self, days_remaining: float):
        """Update time to expiration as sub-heat progresses"""
        self.time_to_expiration = days_remaining / TRADING_YEAR_DAYS
    
    def update_rtm_price(self, new_price: float):
        """Update RTM price"""
        self.rtm_price = new_price
    
    def parse_ticker(self, ticker: str) -> Tuple[bool, float]:
        """
        Parse option ticker to extract type and strike.
        
        Returns:
            (is_call, strike_price)
        """
        # Format: RTM1C50 or RTM1P50
        is_call = 'C' in ticker
        strike = float(ticker.split('C' if is_call else 'P')[1])
        return is_call, strike
    
    def calculate_implied_volatility(self, ticker: str, market_price: float) -> Optional[float]:
        """Calculate implied volatility for an option"""
        is_call, strike = self.parse_ticker(ticker)
        
        return self.bs_calc.implied_volatility(
            market_price=market_price,
            S=self.rtm_price,
            K=strike,
            T=self.time_to_expiration,
            r=RISK_FREE_RATE,
            is_call=is_call
        )
    
    def calculate_fair_value(self, ticker: str, realized_vol: float) -> float:
        """Calculate fair value of option using realized volatility"""
        is_call, strike = self.parse_ticker(ticker)
        
        if is_call:
            return self.bs_calc.call_price(
                S=self.rtm_price,
                K=strike,
                T=self.time_to_expiration,
                r=RISK_FREE_RATE,
                sigma=realized_vol
            )
        else:
            return self.bs_calc.put_price(
                S=self.rtm_price,
                K=strike,
                T=self.time_to_expiration,
                r=RISK_FREE_RATE,
                sigma=realized_vol
            )
    
    def calculate_delta(self, ticker: str, volatility: float) -> float:
        """Calculate option delta"""
        is_call, strike = self.parse_ticker(ticker)
        
        if is_call:
            return self.bs_calc.call_delta(
                S=self.rtm_price,
                K=strike,
                T=self.time_to_expiration,
                r=RISK_FREE_RATE,
                sigma=volatility
            )
        else:
            return self.bs_calc.put_delta(
                S=self.rtm_price,
                K=strike,
                T=self.time_to_expiration,
                r=RISK_FREE_RATE,
                sigma=volatility
            )
    
    def identify_mispricings(self, option_prices: Dict[str, float], 
                            realized_vol: float, 
                            min_edge: float = 0.05) -> List[TradeSignal]:
        """
        Identify mispriced options.
        
        Args:
            option_prices: Dict mapping ticker to market price
            realized_vol: True/realized volatility from analysts
            min_edge: Minimum edge (fair_value - market_price) to trade
        
        Returns:
            List of TradeSignal objects
        """
        signals = []
        
        for ticker, market_price in option_prices.items():
            # Calculate implied volatility from market price
            implied_vol = self.calculate_implied_volatility(ticker, market_price)
            
            if implied_vol is None:
                continue  # Skip if can't calculate IV
            
            # Calculate fair value using realized volatility
            fair_value = self.calculate_fair_value(ticker, realized_vol)
            
            # Calculate edge
            edge = fair_value - market_price
            
            # Determine action
            if abs(edge) < min_edge:
                continue  # Not mispriced enough
            
            action = 'BUY' if edge > 0 else 'SELL'
            
            # Calculate delta for this option
            delta = self.calculate_delta(ticker, realized_vol)
            
            signals.append(TradeSignal(
                ticker=ticker,
                action=action,
                quantity=10,  # Start with 10 contracts as default
                market_price=market_price,
                fair_value=fair_value,
                edge=edge,
                implied_vol=implied_vol,
                realized_vol=realized_vol,
                delta=delta
            ))
        
        # Sort by absolute edge (most mispriced first)
        signals.sort(key=lambda x: abs(x.edge), reverse=True)
        
        return signals
    
    def calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta"""
        option_delta = sum(pos.quantity * pos.delta * OPTION_MULTIPLIER for pos in self.positions)
        rtm_delta = self.rtm_position * 1.0  # RTM shares have delta of 1
        
        return option_delta + rtm_delta
    
    def calculate_hedge_trade(self, target_delta: float = 0.0) -> Tuple[str, int]:
        """
        Calculate RTM trade needed to reach target delta.
        
        Returns:
            (action, quantity) where action is 'BUY' or 'SELL'
        """
        current_delta = self.calculate_portfolio_delta()
        delta_adjustment = target_delta - current_delta
        
        # Round to nearest share
        shares_to_trade = round(delta_adjustment)
        
        if shares_to_trade > 0:
            return 'BUY', shares_to_trade
        elif shares_to_trade < 0:
            return 'SELL', abs(shares_to_trade)
        else:
            return 'HOLD', 0
    
    def execute_trade(self, signal: TradeSignal) -> OptionPosition:
        """Execute an option trade via the API and record the resulting position."""
        self.tapi.post_order(signal.ticker, "MARKET", signal.quantity, signal.action)

        is_call, strike = self.parse_ticker(signal.ticker)
        quantity = signal.quantity if signal.action == 'BUY' else -signal.quantity

        position = OptionPosition(
            ticker=signal.ticker,
            quantity=quantity,
            strike=strike,
            is_call=is_call,
            delta=signal.delta,
            entry_price=signal.market_price
        )

        self.positions.append(position)

        print(f"EXECUTED: {signal.action} {signal.quantity} contracts of {signal.ticker} @ ${signal.market_price:.2f}")
        print(f"  Edge: ${signal.edge:.2f}, IV: {signal.implied_vol:.1%}, RV: {signal.realized_vol:.1%}")

        return position
    
    def execute_hedge(self, action: str, quantity: int):
        """Execute RTM hedge trade via the API."""
        if action == 'HOLD' or quantity == 0:
            return

        if action == 'BUY':
            self.tapi.buy("RTM1", quantity)
            self.rtm_position += quantity
        else:  # SELL
            self.tapi.sell("RTM1", quantity)
            self.rtm_position -= quantity

        print(f"HEDGE: {action} {quantity} shares of RTM1 @ ${self.rtm_price:.2f}")
        print(f"  New RTM position: {self.rtm_position} shares")

    def sync_state(self) -> Dict[str, float]:
        """
        Fetch current market state from the API and update trader internals.
        Returns the option price dict (bid/ask midpoint per ticker).
        """
        case = self.tapi.get_case()
        ticks_left = case["ticks_per_period"] - case["tick"]
        self.time_to_expiration = (ticks_left / case["ticks_per_period"]) * (MONTH_DAYS / TRADING_YEAR_DAYS)

        securities = self.tapi.get_securities()
        prices: Dict[str, float] = {}
        new_positions: List[OptionPosition] = []

        for sec in securities:
            ticker = sec["ticker"]
            bid, ask, last = sec.get("bid"), sec.get("ask"), sec.get("last")
            mid = (bid + ask) / 2 if bid and ask else (last or 0.0)

            if ticker == "RTM1":
                self.rtm_price = mid
                self.rtm_position = int(sec.get("position", 0))
            elif ticker in ALL_OPTION_TICKERS:
                prices[ticker] = mid
                qty = int(sec.get("position", 0))
                if qty != 0:
                    is_call, strike = self.parse_ticker(ticker)
                    delta = self.calculate_delta(ticker, self.last_realized_vol)
                    new_positions.append(OptionPosition(
                        ticker=ticker, quantity=qty, strike=strike,
                        is_call=is_call, delta=delta, entry_price=mid
                    ))

        self.positions = new_positions
        return prices

    def check_limits(self) -> bool:
        """Return True if there is headroom to add more positions (5% safety buffer)."""
        for lim in self.tapi.get_limits():
            if lim["gross"] >= lim["gross_limit"] * 0.95:
                return False
            if abs(lim["net"]) >= lim["net_limit"] * 0.95:
                return False
        return True

    def calc_order_quantity(self) -> int:
        """
        Return per-trade contract size that spreads remaining net headroom evenly
        across all option tickers, leaving room for future trades.
        """
        min_qty = float("inf")
        for lim in self.tapi.get_limits():
            headroom = lim["net_limit"] * 0.95 - abs(lim["net"])
            per_trade = max(1, int(headroom // len(ALL_OPTION_TICKERS)))
            min_qty = min(min_qty, per_trade)
        return int(min_qty) if min_qty != float("inf") else 10

    def close_corrected_positions(self, option_prices: Dict[str, float],
                                  realized_vol: float, close_threshold: float = 0.03):
        """
        Close positions where the mispricing has been corrected by the market maker.
        - Long positions: close when market price has risen (edge <= close_threshold)
        - Short positions: close when market price has fallen (edge >= -close_threshold)
        """
        for pos in list(self.positions):
            market_price = option_prices.get(pos.ticker)
            if market_price is None:
                continue
            fair_value = self.calculate_fair_value(pos.ticker, realized_vol)
            edge = fair_value - market_price
            if pos.quantity > 0 and edge <= close_threshold:
                self.tapi.post_order(pos.ticker, "MARKET", abs(pos.quantity), "SELL")
                print(f"CLOSE LONG:  {pos.ticker} @ ${market_price:.2f} (fair=${fair_value:.2f}, edge=${edge:.2f})")
            elif pos.quantity < 0 and edge >= -close_threshold:
                self.tapi.post_order(pos.ticker, "MARKET", abs(pos.quantity), "BUY")
                print(f"CLOSE SHORT: {pos.ticker} @ ${market_price:.2f} (fair=${fair_value:.2f}, edge=${edge:.2f})")


def poll_option_prices(tapi: TradingAPI) -> Dict[str, float]:
    """
    Poll option prices from the RIT securities API.
    Uses bid/ask midpoint as market price.

    Returns:
        Dict mapping option ticker to market price
    """
    securities = tapi.get_securities()
    prices = {}
    for sec in securities:
        ticker = sec["ticker"]
        if ticker not in ALL_OPTION_TICKERS:
            continue
        bid = sec.get("bid")
        ask = sec.get("ask")
        if bid and ask:
            prices[ticker] = (bid + ask) / 2
        elif sec.get("last"):
            prices[ticker] = sec["last"]
    return prices


def get_analyst_vol() -> float:
    """
    Dummy function to get analyst forecasted volatility.
    Replace with actual news feed parser.
    
    Returns:
        Realized volatility as a decimal (e.g., 0.35 for 35%)
    """
    # Example - in reality, parse from RIT news feed
    return 0.35  # 35% volatility


def run_trading_loop():
    """Main polling loop: sync state, close corrected positions, open new trades, hedge, sleep."""
    import time

    # rtm_price and time_to_expiration will be overwritten by sync_state() on the first tick
    trader = VolatilityArbitrageTrader(rtm_price=50.0, time_to_expiration_days=20)

    print("=" * 60)
    print("VOLATILITY ARBITRAGE TRADER — live loop started")
    print("=" * 60)

    while True:
        try:
            # 1. Sync market state and get current option prices
            option_prices = trader.sync_state()

            # 2. Get analyst volatility forecast
            realized_vol = get_analyst_vol()
            trader.last_realized_vol = realized_vol

            print(f"\n[tick] RTM1=${trader.rtm_price:.2f}  RV={realized_vol:.1%}"
                  f"  T={trader.time_to_expiration * TRADING_YEAR_DAYS:.1f}d"
                  f"  positions={len(trader.positions)}")

            # 3. Close positions where mispricing has been corrected
            trader.close_corrected_positions(option_prices, realized_vol)

            # 4. Re-hedge after any closes
            action, qty = trader.calculate_hedge_trade(target_delta=0.0)
            trader.execute_hedge(action, qty)

            # 5. Identify and open new mispriced positions if within limits
            if trader.check_limits():
                signals = trader.identify_mispricings(option_prices, realized_vol, min_edge=0.10)
                order_qty = trader.calc_order_quantity()
                for signal in signals:
                    if not trader.check_limits():
                        break
                    signal.quantity = order_qty
                    trader.execute_trade(signal)

            # 6. Re-hedge after new positions
            action, qty = trader.calculate_hedge_trade(target_delta=0.0)
            trader.execute_hedge(action, qty)

        except Exception as e:
            print(f"Loop error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    run_trading_loop()