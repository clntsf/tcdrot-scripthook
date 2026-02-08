# systematic implementation of a butterfly spread
# currently template until I work out how to do the API for real

from sys import argv
from api import TradingAPI

tapi = TradingAPI()

def butterfly(t1: int, t2: int, n=1):
    if (t1 + t2) %2 != 0:
        print("[ERR] midpoint not whole number, cannot transact!")
        return

    tickt1 = f"RTM1C{t1}"
    tickt2 = f"RTM1C{t2}"

    mid = (t1+t2)//2
    tickmid = f"RTM1P{mid}"

    tapi.buy(tickt1, n, t1)
    tapi.buy(tickt2, n, t2)
    tapi.sell(tickmid, 2*n, mid)

if __name__ == "__main__":
    t1 = int(argv[1])
    t2 = int(argv[2])
    n = 1 if len(argv) < 5 else int(argv[4])
    butterfly(t1,t2,n)