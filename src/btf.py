# systematic implementation of a butterfly spread
# currently template until I work out how to do the API for real

from sys import argv
from api import TradingAPI

tapi = TradingAPI()

def butterfly(ticker, t1, t2, n=1):
    tapi.buy(ticker, n, t1)
    tapi.buy(ticker, n, t2)
    tapi.sell(ticker, 2*n, (t1+t2)/2)

if __name__ == "__main__":
    ticker = argv[1]
    t1 = float(argv[2])
    t2 = float(argv[3])
    n = 1 if len(argv) < 5 else int(argv[4])
    butterfly(ticker,t1,t2,n)