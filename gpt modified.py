"""
RIT Market Simulator Momentum Trading Bot
Rotman International Trading Competition (RITC)
Modified for: Momentum Strategy + Per-Stock Gross/Net Limits
"""

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import matplotlib.pyplot as plt

# ========= CONFIG =========
API = "http://localhost:9999/v1"
API_KEY = "Rotman"
HDRS = {"X-API-key": API_KEY}

NGN, WHEL, GEAR, RSM1000 = "NGN", "WHEL", "GEAR", "RSM1000"

# Trading limits
MAX_TRADE_SIZE       = 10_000          # Maximum order size
PER_STOCK_GROSS_LIMIT = 40_000          # Total long+short limit per ticker
PER_STOCK_NET_LIMIT   = 20_000          # Max directional limit per ticker

SLEEP_SEC = 0.25
PRINT_HEARTBEAT = True

# ========= SESSION =========
s = requests.Session()
s.headers.update(HDRS)

# ========= BASIC HELPERS =========
def get_tick_status():
    r = s.get(f"{API}/case")
    r.raise_for_status()
    j = r.json()
    return j["tick"], j["status"]

def best_bid_ask(ticker):
    r = s.get(f"{API}/securities/book", params={"ticker": ticker})
    r.raise_for_status()
    book = r.json()
    bid = float(book["bids"][0]["price"]) if book["bids"] else 0.0
    ask = float(book["asks"][0]["price"]) if book["asks"] else 1e12
    return bid, ask

def mid_price(ticker):
    bid, ask = best_bid_ask(ticker)
    if bid == 0.0 and ask == 1e12:
        return None
    return 0.5 * (bid + ask)

def positions_map():
    r = s.get(f"{API}/securities")
    r.raise_for_status()
    out = {p["ticker"]: int(p.get("position", 0)) for p in r.json()}
    for k in (NGN, WHEL, GEAR, RSM1000):
        out.setdefault(k, 0)
    return out

def place_mkt(ticker, action, qty):
    qty = int(max(1, min(qty, MAX_TRADE_SIZE)))
    r = s.post(f"{API}/orders",
               params={"ticker": ticker, "type": "MARKET",
                       "quantity": qty, "action": action})
    if PRINT_HEARTBEAT:
        print(f"ORDER {action} {qty} {ticker} -> {'OK' if r.ok else 'FAIL'}")
    return r.ok

# ========= STOCK LIMIT CHECK =========
def within_limits_stock(stk, qty, action):
    pos = positions_map()[stk]

    if action == "BUY":
        new_pos = pos + qty
    else:
        new_pos = pos - qty

    # Gross = |position|
    if abs(new_pos) > PER_STOCK_GROSS_LIMIT:
        return False

    # Net directional limit
    if abs(new_pos) > PER_STOCK_NET_LIMIT:
        return False

    return True

# ========= HISTORICAL LOADING =========
def load_historical():
    r = s.get(f"{API}/news")
    r.raise_for_status()
    news = r.json()
    if not news:
        print("No historical table found.")
        return None

    soup = BeautifulSoup(news[0].get("body", ""), "html.parser")
    table = soup.find("table")
    if not table:
        print("Table missing.")
        return None

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) == 5:
            rows.append(cols)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["Tick"] = df["Tick"].astype(int)
    for c in ["RSM1000", "NGN", "WHEL", "GEAR"]:
        df[c] = df[c].astype(float)
    return df

def print_three_tables_and_betas(df_hist):
    returns = df_hist[["RSM1000", "NGN", "WHEL", "GEAR"]].pct_change().dropna()
    idx_var = returns["RSM1000"].var()
    beta_map = {
        t: float(np.cov(returns[t], returns["RSM1000"])[0,1] / idx_var)
        for t in ["RSM1000","NGN","WHEL","GEAR"]
    }
    print("\nLoaded historical table + betas.\n")
    return beta_map

# ========= MAIN MOMENTUM STRATEGY =========
def main():
    df_hist = load_historical()
    if df_hist is None:
        return
    print_three_tables_and_betas(df_hist)

    # EMA setup
    MA_DAYS = 30
    MOMENTUM_LOOKBACK = 12
    MOMENTUM_ENTRY = 0.00010
    MOMENTUM_EXIT  = 0.00002
    MIN_ORDER = 2000

    ma_list = {
        NGN: [mid_price(NGN)] * 20,
        WHEL: [mid_price(WHEL)] * 20,
        GEAR: [mid_price(GEAR)] * 20
    }

    ma = {NGN: 0, WHEL: 0, GEAR: 0}

    def moving_avg(stk, prev):
        price = mid_price(stk)
        if price is None:
            return prev
        alpha = 2 / (MA_DAYS + 1)
        return alpha * price + (1 - alpha) * prev

    def calc_momentum(stk):
        if len(ma_list[stk]) < MOMENTUM_LOOKBACK + 2:
            return 0
        recent = ma_list[stk][-MOMENTUM_LOOKBACK:]
        y = np.array(recent)
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0] / y[-1]  # normalized slope
        return slope

    def dynamic_order_size(stk, momentum):
        BASE = 5000
        curr_pos = abs(positions_map()[stk])

        # shrink size near limit
        limit_factor = max(0.10, 1 - curr_pos / PER_STOCK_GROSS_LIMIT)

        # grow size with momentum
        mom_factor = min(2.0, 1 + 12 * abs(momentum))

        size = int(BASE * limit_factor * mom_factor)

        # enforce remaining room
        remaining = PER_STOCK_GROSS_LIMIT - curr_pos
        size = min(size, remaining, MAX_TRADE_SIZE)

        return max(MIN_ORDER, size)

    def trade_momentum(stk, price):
        curr_ma = ma[stk]
        mom = calc_momentum(stk)
        pos = positions_map()[stk]

        # BUY signal
        if price > curr_ma and mom > MOMENTUM_ENTRY:
            size = dynamic_order_size(stk, mom)
            if within_limits_stock(stk, size, "BUY"):
                place_mkt(stk, "BUY", size)
            return

        # SELL signal
        if price < curr_ma and mom < -MOMENTUM_ENTRY:
            size = dynamic_order_size(stk, mom)
            if within_limits_stock(stk, size, "SELL"):
                place_mkt(stk, "SELL", size)
            return

        # EXIT / FLATTEN when momentum weak
        if abs(mom) < MOMENTUM_EXIT:
            if pos > 0:
                qty = min(pos, PER_STOCK_GROSS_LIMIT)
                if within_limits_stock(stk, qty, "SELL"):
                    place_mkt(stk, "SELL", qty)

            elif pos < 0:
                qty = min(abs(pos), PER_STOCK_GROSS_LIMIT)
                if within_limits_stock(stk, qty, "BUY"):
                    place_mkt(stk, "BUY", qty)

    # Main loop
    tick, status = get_tick_status()
    while status == "ACTIVE":

        mids = {
            NGN: mid_price(NGN),
            WHEL: mid_price(WHEL),
            GEAR: mid_price(GEAR)
        }

        for stk in [NGN, WHEL, GEAR]:
            price = mids[stk]
            if price is None:
                continue

            # update moving average
            new_ma = moving_avg(stk, ma_list[stk][-1])
            ma_list[stk].append(new_ma)
            ma[stk] = new_ma

            # execute strategy
            trade_momentum(stk, price)

        sleep(SLEEP_SEC)
        tick, status = get_tick_status()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
