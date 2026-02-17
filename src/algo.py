"""RITC 2026 Algorithmic Market Making Trading Case - REST API Basic Script"""

import requests
from time import sleep
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
API_PORT = getenv("ROT_API_PORT")

s = requests.Session()
s.headers.update({'X-API-key': API_KEY})

# ANSI colour codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Sample setup
MAX_EXPOSURE = 15000
ORDER_LIMIT = 500

def get_tick():
    resp = s.get(f'http://localhost:{API_PORT}/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick'], case['status']

def get_bid_ask(ticker):
    payload = {'ticker': ticker}
    resp = s.get (f'http://localhost:{API_PORT}/v1/securities/book', params = payload)
    if resp.ok:
        book = resp.json()
        bid_side_book = book['bids']
        ask_side_book = book['asks']

        # Check if book is empty
        if not bid_side_book or not ask_side_book:
            return None, None
        
        bid_prices_book = [item["price"] for item in bid_side_book]
        ask_prices_book = [item['price'] for item in ask_side_book]
        
        best_bid_price = bid_prices_book[0]
        best_ask_price = ask_prices_book[0]
  
        return best_bid_price, best_ask_price
    return None, None

def get_time_sales(ticker):
    payload = {'ticker': ticker}
    resp = s.get (f'http://localhost:{API_PORT}/v1/securities/tas', params = payload)
    if resp.ok:
        book = resp.json()
        time_sales_book = [item["quantity"] for item in book]
        return time_sales_book

def get_ind_position(ticker):
    payload = {'ticker': ticker}
    resp = s.get(f'http://localhost:{API_PORT}/v1/securities', params=payload)
    if resp.ok:
        securities = resp.json()
        for security in securities:
            if security['ticker'] == ticker:
                return security['position']
    return 0  # Return 0 if ticker not found or request fails

def get_position():
    resp = s.get (f'http://localhost:{API_PORT}/v1/securities')
    if resp.ok:
        book = resp.json()
        return abs(book[0]['position']) + abs(book[1]['position']) + abs(book[2]['position']) + abs(book[3]['position'])

def get_open_orders(ticker):
    payload = {'ticker': ticker}
    resp = s.get (f'http://localhost:{API_PORT}/v1/orders', params = payload)
    if resp.ok:
        orders = resp.json()
        buy_orders = [item for item in orders if item["action"] == "BUY"]
        sell_orders = [item for item in orders if item["action"] == "SELL"]
        return buy_orders, sell_orders

def get_order_status(order_id):
    resp = s.get (f'http://localhost:{API_PORT}/v1/orders' + '/' + str(order_id))
    if resp.ok:
        order = resp.json()
        return order['status']

def main():
    print()
    print(f"  {BOLD}RITC Algorithmic Market Maker{RESET}")
    print(f"  MAX_EXPOSURE={MAX_EXPOSURE}  ORDER_LIMIT={ORDER_LIMIT}")
    print(f"  {DIM}Ctrl+C to stop{RESET}")
    print()

    tick, status = get_tick()
    if status != 'ACTIVE':
        print(f"  {YELLOW}Case not active (status={status}). Waiting...{RESET}")
        while status != 'ACTIVE':
            sleep(1)
            tick, status = get_tick()

    ticker_list = [i['ticker'] for i in s.get(f'http://localhost:{API_PORT}/v1/securities').json()]
    print(f"  Tickers: {', '.join(ticker_list)}")
    print(f"  {GREEN}Case ACTIVE — starting at tick {tick}{RESET}")
    print()

    try:
        while status == 'ACTIVE':

            gross_exposure = get_position()
            exposure_pct = (gross_exposure / MAX_EXPOSURE) * 100 if MAX_EXPOSURE else 0
            if exposure_pct >= 90:
                exp_colour = RED
            elif exposure_pct >= 70:
                exp_colour = YELLOW
            else:
                exp_colour = GREEN

            # Position close window: ticks 55-59 of each minute
            if tick % 60 >= 55:
                print(f"  {RED}{BOLD}[tick {tick}] CLOSING WINDOW — flattening all positions{RESET}")
                for ticker_symbol in ticker_list:
                    position = get_ind_position(ticker_symbol)
                    if position > 0:
                        s.post(f'http://localhost:{API_PORT}/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': position, 'action': 'SELL'})
                        print(f"    SELL {position} {ticker_symbol} @ MKT")
                    elif position < 0:
                        s.post(f'http://localhost:{API_PORT}/v1/orders', params={'ticker': ticker_symbol, 'type': 'MARKET', 'quantity': abs(position), 'action': 'BUY'})
                        print(f"    BUY {abs(position)} {ticker_symbol} @ MKT")
            else:
                # Normal market-making tick
                header = (f"  {DIM}[tick {tick}]{RESET}"
                          f"  exposure: {exp_colour}{gross_exposure}/{MAX_EXPOSURE} ({exposure_pct:.0f}%){RESET}")
                print(header)

                for ticker_symbol in ticker_list:
                    best_bid_price, best_ask_price = get_bid_ask(ticker_symbol)

                    if best_bid_price is None or best_ask_price is None:
                        print(f"    {ticker_symbol}: {YELLOW}no book{RESET}")
                        continue

                    spread = best_ask_price - best_bid_price
                    ind_pos = get_ind_position(ticker_symbol)

                    if gross_exposure < MAX_EXPOSURE:
                        s.post(f'http://localhost:{API_PORT}/v1/orders', params={'ticker': ticker_symbol, 'type': 'LIMIT', 'quantity': ORDER_LIMIT, 'price': best_bid_price, 'action': 'BUY'})
                        s.post(f'http://localhost:{API_PORT}/v1/orders', params={'ticker': ticker_symbol, 'type': 'LIMIT', 'quantity': ORDER_LIMIT, 'price': best_ask_price, 'action': 'SELL'})
                        action_str = f"{GREEN}quoted {ORDER_LIMIT} @ {best_bid_price:.2f}/{best_ask_price:.2f}{RESET}"
                    else:
                        action_str = f"{RED}at limit — skipped{RESET}"

                    print(f"    {ticker_symbol:<6} bid={best_bid_price:.2f} ask={best_ask_price:.2f}"
                          f" spd={spread:.2f} pos={ind_pos:>+6d}  {action_str}")

                    sleep(0.5)
                    s.post(f'http://localhost:{API_PORT}/v1/commands/cancel', params={'ticker': ticker_symbol})

            tick, status = get_tick()

    except KeyboardInterrupt:
        print(f"\n  {BOLD}Shutting down.{RESET}")

    print(f"  {DIM}Final status: {status}, tick {tick}{RESET}")


if __name__ == '__main__':
    main()
