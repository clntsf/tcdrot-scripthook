# Trading API interface
# currently dummied to a comical extent

class TradingAPI:

    def buy(self, ticker: str, quantity: int, price):
        print(f"long {quantity}x {ticker} @ {price:.2f}")

    def sell(self, ticker: str, quantity: int, price):
        print(f"short {quantity}x {ticker} @ {price:.2f}")