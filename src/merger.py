#!/usr/bin/env python3
import argparse
import csv
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import requests
from joblib import load

API_KEY_DEFAULT = "8NKMTW4W"
BASE_URL = "http://localhost:9999/v1"
MAX_ORDER_SIZE = 5000
COMMISSION = 0.02


class Colors:
    def __init__(self, enabled=True):
        if enabled:
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.RED = "\033[91m"
            self.GREEN = "\033[92m"
            self.YELLOW = "\033[93m"
            self.CYAN = "\033[96m"
            self.DIM = "\033[2m"
            self.BG_GREEN = "\033[42m"
            self.BG_RED = "\033[41m"
        else:
            self.RESET = self.BOLD = self.RED = self.GREEN = ""
            self.YELLOW = self.CYAN = self.DIM = ""
            self.BG_GREEN = self.BG_RED = ""


def locked_print(lock, *args, **kwargs):
    with lock:
        print(*args, **kwargs)


def locked_input(lock, prompt=""):
    with lock:
        print(prompt, end="", flush=True)
    return input()


DEALS = {
    "D1": {
        "name": "Targenix / Pharmaco",
        "industry": "Pharmaceuticals",
        "target": "TGX",
        "acquirer": "PHR",
        "structure": "cash",
        "cash_component": 50.0,
        "stock_ratio": 0.0,
        "target_start": 43.70,
        "acquirer_start": 47.50,
        "p0": 0.70,
        "deal_multiplier": 1.00,
        "names": ["targenix", "pharmaco"],
        "industry_kw": ["pharma", "pharmaceutical", "drug", "biotech", "fda", "trial", "patent"],
    },
    "D2": {
        "name": "ByteLayer / CloudSys",
        "industry": "Cloud Software",
        "target": "BYL",
        "acquirer": "CLD",
        "structure": "stock",
        "cash_component": 0.0,
        "stock_ratio": 0.75,
        "target_start": 43.50,
        "acquirer_start": 79.30,
        "p0": 0.55,
        "deal_multiplier": 1.05,
        "names": ["bytelayer", "byte layer", "cloudsys", "cloud sys"],
        "industry_kw": ["cloud", "software", "saas", "tech", "data center", "platform", "subscription"],
    },
    "D3": {
        "name": "GreenGrid / PetroNorth",
        "industry": "Energy / Infrastructure",
        "target": "GGD",
        "acquirer": "PNR",
        "structure": "mixed",
        "cash_component": 33.0,
        "stock_ratio": 0.20,
        "target_start": 31.50,
        "acquirer_start": 59.80,
        "p0": 0.50,
        "deal_multiplier": 1.10,
        "names": ["greengrid", "green grid", "petronorth", "petro north"],
        "industry_kw": ["energy", "infrastructure", "grid", "pipeline", "utility", "power", "oil", "gas"],
    },
    "D4": {
        "name": "FinSure / Atlas Bank",
        "industry": "Banking",
        "target": "FSR",
        "acquirer": "ATB",
        "structure": "cash",
        "cash_component": 40.0,
        "stock_ratio": 0.0,
        "target_start": 30.50,
        "acquirer_start": 62.20,
        "p0": 0.38,
        "deal_multiplier": 1.30,
        "names": ["finsure", "fin sure", "atlas bank", "atlas"],
        "industry_kw": ["bank", "banking", "financial", "deposit", "capital ratio", "basel", "credit union"],
    },
    "D5": {
        "name": "SolarPeak / EastEnergy",
        "industry": "Renewable Energy",
        "target": "SPK",
        "acquirer": "EEC",
        "structure": "stock",
        "cash_component": 0.0,
        "stock_ratio": 1.20,
        "target_start": 52.80,
        "acquirer_start": 48.00,
        "p0": 0.45,
        "deal_multiplier": 1.15,
        "names": ["solarpeak", "solar peak", "eastenergy", "east energy"],
        "industry_kw": ["renewable", "solar", "wind", "green energy", "clean energy", "carbon", "esg"],
    },
}

CATEGORY_LABELS = {
    "REG": "REGULATORY",
    "FIN": "FINANCIAL",
    "PRC": "PRICE/DEAL",
    "SHR": "SHAREHOLDER",
    "ALT": "ALTERNATIVE",
}

CATEGORY_MULTIPLIERS = {"REG": 1.25, "FIN": 1.00, "SHR": 0.90, "ALT": 1.40, "PRC": 0.70}
BASE_IMPACTS = {
    ("Positive", "Small"): +0.03,
    ("Positive", "Medium"): +0.07,
    ("Positive", "Large"): +0.14,
    ("Negative", "Small"): -0.04,
    ("Negative", "Medium"): -0.09,
    ("Negative", "Large"): -0.18,
    ("Ambiguous", "Small"): 0.00,
    ("Ambiguous", "Medium"): 0.00,
    ("Ambiguous", "Large"): 0.00,
}
TICKER_TO_DEAL = {v["target"].upper(): k for k, v in DEALS.items()} | {
    v["acquirer"].upper(): k for k, v in DEALS.items()
}
DEAL_IDS = tuple(DEALS.keys())


def _installed_version(dist_name):
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "not installed"
    except Exception:
        return "unknown"


def _parse_semver(ver):
    if not ver or ver in ("unknown", "not installed"):
        return None
    nums = re.findall(r"\d+", ver)
    if len(nums) < 2:
        return None
    return (int(nums[0]), int(nums[1]), int(nums[2]) if len(nums) > 2 else 0)


def _read_model_sklearn_version(model_path):
    try:
        blob = open(model_path, "rb").read()
    except OSError:
        return None
    m = re.search(rb"_sklearn_version.{0,32}?([0-9]+\.[0-9]+\.[0-9]+)", blob, flags=re.DOTALL)
    if not m:
        return None
    try:
        return m.group(1).decode("ascii")
    except Exception:
        return None


def _format_env_error(model_path, model_sklearn_version, reasons):
    numpy_v = _installed_version("numpy")
    scipy_v = _installed_version("scipy")
    sklearn_v = _installed_version("scikit-learn")
    joblib_v = _installed_version("joblib")
    lines = [
        "",
        "  Model environment is incompatible for loading this file.",
        f"  Model: {model_path}",
        f"  Installed: numpy={numpy_v}, scipy={scipy_v}, scikit-learn={sklearn_v}, joblib={joblib_v}",
    ]
    if model_sklearn_version:
        lines.append(f"  Model metadata: _sklearn_version={model_sklearn_version}")
    lines += ["", "  Problems detected:"]
    lines += [f"    - {r}" for r in reasons]
    lines += [
        "",
        "  Fix (Anaconda base env):",
        "    C:\\Users\\andre\\anaconda3\\python.exe -m pip install --upgrade \"scipy>=1.16\" \"scikit-learn==1.8.0\" \"joblib>=1.4\"",
        "",
        "  Safer fix (new env):",
        "    conda create -n rit-news python=3.11 -y",
        "    conda activate rit-news",
        "    python -m pip install requests pandas numpy scipy scikit-learn==1.8.0 joblib",
        "",
    ]
    return "\n".join(lines)


def _validate_model_environment(model_path):
    reasons = []
    numpy_v = _installed_version("numpy")
    scipy_v = _installed_version("scipy")
    sklearn_v = _installed_version("scikit-learn")
    model_sklearn_v = _read_model_sklearn_version(model_path)
    numpy_sem, scipy_sem = _parse_semver(numpy_v), _parse_semver(scipy_v)
    sklearn_sem = _parse_semver(sklearn_v)
    model_sem = _parse_semver(model_sklearn_v) if model_sklearn_v else None
    if numpy_sem and scipy_sem and numpy_sem[0] >= 2 and scipy_sem < (1, 13, 0):
        reasons.append(
            f"numpy {numpy_v} with scipy {scipy_v} is an ABI mismatch (SciPy < 1.13 was built for NumPy 1.x)."
        )
    if model_sem and sklearn_sem and (model_sem[0], model_sem[1]) != (sklearn_sem[0], sklearn_sem[1]):
        reasons.append(f"model was trained with scikit-learn {model_sklearn_v}, but installed version is {sklearn_v}.")
    if reasons:
        raise RuntimeError(_format_env_error(model_path, model_sklearn_v, reasons))

class NewsClassifier:
    def __init__(self, model_path, dir_threshold=0.70, cat_threshold=0.70):
        _validate_model_environment(model_path)
        try:
            saved = load(model_path)
        except Exception as exc:
            raise RuntimeError(f"\n  Failed to load model '{model_path}': {type(exc).__name__}: {exc}\n") from exc
        self.models = saved["models"]
        self.metadata = saved.get("metadata", {})
        self.dir_threshold = dir_threshold
        self.cat_threshold = cat_threshold

    def classify(self, headline, body):
        text_input = [f"{headline} | {body}"]
        out = {}
        for target, pipeline in self.models.items():
            prediction = pipeline.predict(text_input)[0]
            clf = pipeline.named_steps["clf"]
            feat = pipeline.named_steps["features"]
            x_t = feat.transform(text_input)
            confidence, class_probs = 0.0, {}
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(x_t)[0]
                confidence = float(max(proba))
                class_probs = dict(zip(clf.classes_, [float(p) for p in proba]))
            elif hasattr(clf, "decision_function"):
                dec = clf.decision_function(x_t)
                if isinstance(dec, np.ndarray):
                    dec = dec[0]
                if isinstance(dec, np.ndarray):
                    exp_d = np.exp(dec - dec.max())
                    proba = exp_d / exp_d.sum()
                    confidence = float(max(proba))
                    class_probs = dict(zip(clf.classes_, [float(p) for p in proba]))
                else:
                    confidence = float(1.0 / (1.0 + np.exp(-abs(float(dec)))))
            out[target] = {"prediction": prediction, "confidence": confidence, "class_probs": class_probs}
        return out


def _word_boundary_match(text, keyword):
    return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE))


def _extract_deal_ids_from_api_ticker(api_ticker):
    raw = str(api_ticker or "").upper().strip()
    if not raw:
        return []
    if raw == "ALL" or "ALL" in re.split(r"[^A-Z0-9]+", raw):
        return ["ALL"]

    deals = []
    for m in re.finditer(r"\bD[1-5]\b", raw):
        d = m.group(0)
        if d in DEALS and d not in deals:
            deals.append(d)

    if deals:
        return deals

    for tok in re.split(r"[^A-Z0-9]+", raw):
        if not tok:
            continue
        d = TICKER_TO_DEAL.get(tok)
        if d and d not in deals:
            deals.append(d)
    return deals


def identify_deal(headline, body="", api_ticker=""):
    text = f"{headline} {body}".strip()
    text_l = text.lower()
    api_tokens = set(re.split(r"[^A-Z0-9]+", str(api_ticker or "").upper()))
    scores = {}
    for deal_id, deal in DEALS.items():
        score = 0
        for ticker in (deal["target"], deal["acquirer"]):
            score += _word_boundary_match(text, ticker) * 5
            if ticker.upper() in api_tokens:
                score += 5
        for name in deal["names"]:
            if name in text_l:
                score += 3
        for kw in deal["industry_kw"]:
            if kw in text_l:
                score += 1
        scores[deal_id] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_id, best_score = ranked[0]
    second_id, second_score = ranked[1]
    if best_score == 0:
        return None, 0, "low", None
    alt = second_id if best_score == second_score else None
    conf = "high" if best_score >= 5 else "medium" if best_score >= 1 else "low"
    if alt:
        conf = "low"
    return best_id, best_score, conf, alt


def resolve_deal(api_ticker, headline, body):
    api_deals = _extract_deal_ids_from_api_ticker(api_ticker)
    if api_deals:
        if api_deals[0] == "ALL":
            return "ALL", "api_ticker_all", "high", 5, None
        if len(api_deals) == 1:
            return api_deals[0], "api_ticker", "high", 5, None
        return api_deals[0], "api_ticker_multi", "medium", 4, api_deals[1]
    deal_id, score, conf, alt = identify_deal(headline, body, api_ticker or "")
    if deal_id and conf == "high":
        return deal_id, "fallback_text", conf, score, alt
    return None, "unmapped", conf, score, alt


def map_impact_to_severity(impact_prediction):
    impact = str(impact_prediction or "").strip()
    return (impact, False) if impact in ("Small", "Medium", "Large") else ("Medium", True)


def get_case_status(session):
    try:
        resp = session.get(f"{BASE_URL}/case", timeout=2)
        if resp.ok:
            d = resp.json()
            return d.get("tick", 0), d.get("status", "UNKNOWN"), d.get("name", "Unknown")
        return 0, "ERROR", "Unknown"
    except requests.ConnectionError:
        return 0, "DISCONNECTED", "Unknown"
    except Exception:
        return 0, "ERROR", "Unknown"


def get_news_items(session, limit):
    try:
        resp = session.get(f"{BASE_URL}/news", params={"limit": limit}, timeout=3)
        if resp.ok:
            d = resp.json()
            return d if isinstance(d, list) else []
        return []
    except Exception:
        return []


class MergerArbEngine:
    def __init__(self, session, colors, print_lock):
        self.session = session
        self.colors = colors
        self.print_lock = print_lock
        self.lock = threading.RLock()
        self.probabilities = {d: info["p0"] for d, info in DEALS.items()}
        self.standalone_values = {}
        self.news_count = {d: 0 for d in DEALS}
        self.initialized = False

    def initialize(self):
        c = self.colors
        lines = ["", f"{c.BOLD}{'=' * 78}{c.RESET}", f"{c.BOLD} MERGER ARB ENGINE - INITIALIZING{c.RESET}"]
        values = {}
        for deal_id, d in DEALS.items():
            p0 = d["p0"]
            p_start = d["target_start"]
            k0 = d["cash_component"] + d["stock_ratio"] * d["acquirer_start"]
            v = (p_start - p0 * k0) / (1 - p0)
            values[deal_id] = v
            fair = p0 * k0 + (1 - p0) * v
            lines.append(f"  {c.CYAN}{deal_id}{c.RESET} {d['name']:<30} K0=${k0:>7.2f} V=${v:>7.2f} p0={p0:.0%} Fair=${fair:>7.2f}")
        with self.lock:
            self.standalone_values = values
            self.initialized = True
        lines += [f"  {c.DIM}Standalone values locked. News updates probabilities only.{c.RESET}", f"{c.BOLD}{'=' * 78}{c.RESET}", ""]
        with self.print_lock:
            print("\n".join(lines))

    def get_probability(self, deal_id):
        with self.lock:
            return self.probabilities[deal_id]

    def set_probability(self, deal_id, new_p):
        with self.lock:
            old = self.probabilities[deal_id]
            self.probabilities[deal_id] = float(new_p)
            return old

    def get_market_price(self, ticker):
        try:
            resp = self.session.get(f"{BASE_URL}/securities", params={"ticker": ticker}, timeout=2)
            if resp.ok:
                d = resp.json()
                if isinstance(d, list) and d:
                    return d[0].get("last", d[0].get("bid"))
                if isinstance(d, dict):
                    return d.get("last", d.get("bid"))
        except Exception:
            pass
        return None

    def get_position(self, ticker):
        try:
            resp = self.session.get(f"{BASE_URL}/securities", params={"ticker": ticker}, timeout=2)
            if resp.ok:
                d = resp.json()
                if isinstance(d, list) and d:
                    return d[0].get("position", 0)
                if isinstance(d, dict):
                    return d.get("position", 0)
        except Exception:
            pass
        return 0

    def get_deal_value(self, deal_id, acquirer_price=None):
        d = DEALS[deal_id]
        if d["structure"] == "cash":
            return d["cash_component"]
        if acquirer_price is None:
            acquirer_price = self.get_market_price(d["acquirer"]) or d["acquirer_start"]
        return d["cash_component"] + d["stock_ratio"] * acquirer_price

    def get_fair_value(self, deal_id, probability=None, acquirer_price=None):
        if probability is None:
            probability = self.get_probability(deal_id)
        with self.lock:
            v = self.standalone_values[deal_id]
        k = self.get_deal_value(deal_id, acquirer_price)
        return probability * k + (1 - probability) * v

    def calc_delta_p(self, deal_id, category, direction, severity):
        return BASE_IMPACTS.get((direction, severity), 0.0) * CATEGORY_MULTIPLIERS.get(category, 1.0) * DEALS[deal_id]["deal_multiplier"]

    def process_signal(self, deal_id, category, direction, severity, headline="", verbose=True):
        if deal_id not in DEALS:
            locked_print(self.print_lock, f"  {self.colors.RED}Unknown deal: {deal_id}{self.colors.RESET}")
            return None
        delta_p = self.calc_delta_p(deal_id, category, direction, severity)
        with self.lock:
            old_p = self.probabilities[deal_id]
            new_p = max(0.0, min(1.0, old_p + delta_p))
            self.probabilities[deal_id] = new_p
            self.news_count[deal_id] += 1
        d = DEALS[deal_id]
        target_mkt = self.get_market_price(d["target"])
        acq_mkt = self.get_market_price(d["acquirer"])
        fair = self.get_fair_value(deal_id, new_p, acq_mkt)
        edge = (fair - target_mkt) if target_mkt is not None else 0.0
        c = self.colors
        if verbose:
            with self.print_lock:
                print("\n" + "=" * 78)
                print(f"  {c.BOLD}{c.CYAN}ENGINE UPDATE: {deal_id} {d['name']}{c.RESET}")
                if headline:
                    print(f"  {c.DIM}{(headline[:90] + '...') if len(headline) > 90 else headline}{c.RESET}")
                print(f"  Classification: {direction} / {severity} / {category}")
                print(f"  Delta p = {delta_p:+.4f} -> p: {old_p:.1%} -> {c.BOLD}{new_p:.1%}{c.RESET}")
                print(f"  Fair=${fair:.2f}  Mkt={f'${target_mkt:.2f}' if target_mkt is not None else 'N/A'}  Edge={edge:+.2f}")
                if target_mkt is not None and abs(edge) > 0.05:
                    action = "BUY" if edge > 0 else "SELL"
                    cmd = f"{'b' if edge > 0 else 's'} {deal_id.lower()} [qty]"
                    print(f"  {c.BOLD}ACTION: {action} {d['target']} -> {cmd}{c.RESET}")
                print("=" * 78 + "\n")
        return {"deal": deal_id, "fair_value": fair, "market_price": target_mkt, "edge": edge, "new_p": new_p, "delta_p": delta_p}

    def get_all_positions(self):
        pos = {}
        for d in DEALS.values():
            pos[d["target"]] = self.get_position(d["target"])
            pos[d["acquirer"]] = self.get_position(d["acquirer"])
        return pos

    def submit_order(self, ticker, action, quantity, order_type="MARKET", price=None):
        if quantity <= 0:
            return True
        ok, remaining = True, quantity
        while remaining > 0:
            chunk = min(remaining, MAX_ORDER_SIZE)
            params = {"ticker": ticker, "type": order_type, "quantity": chunk, "action": action}
            if order_type == "LIMIT" and price is not None:
                params["price"] = round(price, 2)
            try:
                resp = self.session.post(f"{BASE_URL}/orders", params=params, timeout=3)
                if resp.ok:
                    locked_print(self.print_lock, f"    {self.colors.GREEN}OK{self.colors.RESET} {action} {chunk} {ticker} @ {order_type}")
                else:
                    ok = False
                    locked_print(self.print_lock, f"    {self.colors.RED}FAIL{self.colors.RESET} {action} {chunk} {ticker}: HTTP {resp.status_code}")
            except Exception as exc:
                ok = False
                locked_print(self.print_lock, f"    {self.colors.RED}ERROR{self.colors.RESET} {action} {chunk} {ticker}: {exc}")
            remaining -= chunk
            if remaining > 0:
                time.sleep(0.1)
        return ok

    def execute_trade(self, deal_id, action, quantity):
        deal_id = deal_id.upper()
        if deal_id not in DEALS:
            locked_print(self.print_lock, f"  {self.colors.RED}Unknown deal: {deal_id}{self.colors.RESET}")
            return
        if quantity <= 0:
            locked_print(self.print_lock, f"  {self.colors.RED}Quantity must be positive.{self.colors.RESET}")
            return
        d = DEALS[deal_id]
        locked_print(self.print_lock, f"\n  {self.colors.BOLD}Executing: {action} {quantity} {d['target']} ({deal_id}){self.colors.RESET}")
        self.submit_order(d["target"], action, quantity)
        if d["structure"] in ("stock", "mixed") and d["stock_ratio"] > 0:
            hedge_qty = int(round(quantity * d["stock_ratio"]))
            hedge_action = "SELL" if action == "BUY" else "BUY"
            if hedge_qty > 0:
                locked_print(self.print_lock, f"  {self.colors.DIM}Auto-hedging: {hedge_action} {hedge_qty} {d['acquirer']}{self.colors.RESET}")
                self.submit_order(d["acquirer"], hedge_action, hedge_qty)
        locked_print(self.print_lock, f"  {self.colors.GREEN}Done.{self.colors.RESET}\n")

    def rehedge_all(self):
        c = self.colors
        with self.print_lock:
            print("\n" + "=" * 78)
            print(f"  {c.BOLD}POSITION & HEDGE STATUS{c.RESET}")
            print("=" * 78)
        pos = self.get_all_positions()
        for deal_id, d in DEALS.items():
            target_pos = pos.get(d["target"], 0)
            acq_pos = pos.get(d["acquirer"], 0)
            ratio = d["stock_ratio"]
            p = self.get_probability(deal_id)
            fair = self.get_fair_value(deal_id, p)
            market = self.get_market_price(d["target"])
            edge = (fair - market) if market is not None else 0.0
            if d["structure"] == "cash":
                hedge_needed, hedge_status = 0, f"{c.DIM}N/A (cash){c.RESET}"
            else:
                ideal = -int(round(target_pos * ratio))
                hedge_needed = ideal - acq_pos
                hedge_status = f"{c.GREEN}Hedged{c.RESET}" if abs(hedge_needed) <= 1 else f"{c.RED}Off by {hedge_needed:+d}{c.RESET}"
            with self.print_lock:
                print(f"\n  {c.CYAN}{deal_id}{c.RESET} p={p:.1%} Fair=${fair:.2f} Edge={edge:+.2f}")
                print(f"       {d['target']}: {target_pos:+6d}  {d['acquirer']}: {acq_pos:+6d}  Hedge: {hedge_status}")
            if d["structure"] != "cash" and abs(hedge_needed) > 1:
                ans = locked_input(self.print_lock, f"       Fix hedge ({hedge_needed:+d} {d['acquirer']})? [y/N]: ").strip().lower()
                if ans == "y":
                    self.submit_order(d["acquirer"], "BUY" if hedge_needed > 0 else "SELL", abs(hedge_needed))
        with self.print_lock:
            print("\n" + "=" * 78 + "\n")

    def close_all(self):
        with self.print_lock:
            print("\n" + "=" * 78)
            print(f"  {self.colors.BOLD}{self.colors.RED}CLOSING ALL POSITIONS{self.colors.RESET}")
            print("=" * 78)
        pos = self.get_all_positions()
        any_pos = False
        for ticker, qty in pos.items():
            if qty == 0:
                continue
            any_pos = True
            self.submit_order(ticker, "SELL" if qty > 0 else "BUY", abs(qty))
        if not any_pos:
            locked_print(self.print_lock, f"  {self.colors.DIM}No open positions.{self.colors.RESET}")
        with self.print_lock:
            print("=" * 78 + "\n")

    def show_status(self):
        c = self.colors
        with self.print_lock:
            print("\n" + "=" * 78)
            print(f"  {c.BOLD}DEAL STATUS{c.RESET}")
            print("-" * 78)
            print(f"  {'Deal':<5} {'Name':<25} {'p':>6} {'Fair':>8} {'Mkt':>8} {'Edge':>8} {'News':>5}")
            print("-" * 78)
        for deal_id, d in DEALS.items():
            with self.lock:
                p, n = self.probabilities[deal_id], self.news_count[deal_id]
            fair = self.get_fair_value(deal_id, p)
            mkt = self.get_market_price(d["target"])
            edge = (fair - mkt) if mkt is not None else 0.0
            with self.print_lock:
                print(f"  {c.CYAN}{deal_id}{c.RESET}   {d['name']:<25} {p:>5.1%}  ${fair:>7.2f}  {(f'${mkt:.2f}' if mkt is not None else 'N/A'):>8}  {(f'${edge:+.2f}' if mkt is not None else 'N/A'):>8}  {n:>5}")
        with self.print_lock:
            print("=" * 78 + "\n")

    def implied_prob(self, deal_id):
        deal_id = deal_id.upper()
        d = DEALS[deal_id]
        mkt = self.get_market_price(d["target"])
        if mkt is None:
            return None
        k = self.get_deal_value(deal_id, self.get_market_price(d["acquirer"]))
        with self.lock:
            v = self.standalone_values[deal_id]
        if abs(k - v) < 0.01:
            return None
        return max(0.0, min(1.0, (mkt - v) / (k - v)))

class NewsMonitor:
    def __init__(self, classifier, engine, session, dir_threshold, cat_threshold, poll_interval, news_limit, colors, print_lock, shutdown_event):
        self.classifier = classifier
        self.engine = engine
        self.session = session
        self.dir_threshold = dir_threshold
        self.cat_threshold = cat_threshold
        self.poll_interval = poll_interval
        self.news_limit = news_limit
        self.colors = colors
        self.print_lock = print_lock
        self.shutdown_event = shutdown_event
        self.seen_ids = set()
        self.log = []
        self.log_lock = threading.RLock()

    def _append_log(self, row):
        with self.log_lock:
            self.log.append(row)

    def _print_direction_alert(self, tick, deal_label, direction, confidence, headline):
        is_buy = direction == "Positive"
        action = "BUY" if is_buy else "SELL"
        fg = self.colors.GREEN if is_buy else self.colors.RED
        bg = self.colors.BG_GREEN if is_buy else self.colors.BG_RED
        cmd_target = deal_label.lower() if deal_label in DEALS else "all"
        cmd = f"{'b' if is_buy else 's'} {cmd_target} [qty]"
        title = f"{action} {deal_label}"

        with self.print_lock:
            print(f"\n{fg}{self.colors.BOLD}{'=' * 78}{self.colors.RESET}")
            print(f"{bg}{self.colors.BOLD}{title:^78}{self.colors.RESET}")
            print(f"{fg}{self.colors.BOLD}{'=' * 78}{self.colors.RESET}")
            print(f"  t={tick}  {action} signal ({confidence:.0%})  cmd: {cmd}")
            print(f"  {(headline[:100] + '...') if len(headline) > 100 else headline}")

    def run(self):
        while not self.shutdown_event.is_set():
            try:
                tick, status, _ = get_case_status(self.session)
                if status in ("DISCONNECTED", "ERROR"):
                    self.shutdown_event.wait(1.0)
                    continue
                for item in get_news_items(self.session, self.news_limit):
                    if self.shutdown_event.is_set():
                        break
                    news_id = item.get("news_id", item.get("id"))
                    headline = item.get("headline", "") or ""
                    body = item.get("body", "") or ""
                    ticker = item.get("ticker", "") or ""
                    if news_id is None:
                        news_id = f"{tick}:{hash(headline + body)}"
                    if news_id in self.seen_ids or not headline.strip():
                        continue
                    self.seen_ids.add(news_id)
                    d, mapping_source, mapping_conf, mapping_score, alt = resolve_deal(ticker, headline, body)
                    resolved_deal, alt_deal = d or "", alt or ""
                    r = self.classifier.classify(headline, body)
                    dir_conf = r["Direction"]["confidence"]
                    cat_conf = r["Category"]["confidence"]
                    direction = r["Direction"]["prediction"]
                    category = r["Category"]["prediction"]

                    actionable_dir = dir_conf >= self.dir_threshold and direction in ("Positive", "Negative")
                    if actionable_dir:
                        tier = "SIGNAL"
                    elif cat_conf >= self.cat_threshold:
                        tier = "CATEGORY"
                    else:
                        tier = "LOW"
                    impact = r.get("Impact", {}).get("prediction", "")
                    impact_conf = r.get("Impact", {}).get("confidence", 0.0)
                    severity, impact_fallback = map_impact_to_severity(impact)
                    engine_updated = False

                    if actionable_dir and (resolved_deal in DEALS or resolved_deal == "ALL"):
                        self._print_direction_alert(tick, resolved_deal, direction, dir_conf, headline)

                    if tier == "SIGNAL" and resolved_deal in DEALS:
                        engine_updated = self.engine.process_signal(
                            resolved_deal, category, direction, severity, headline=headline, verbose=False
                        ) is not None

                    self._append_log(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "tick": tick,
                            "news_id": news_id,
                            "ticker": ticker,
                            "resolved_deal": resolved_deal,
                            "mapping_source": mapping_source,
                            "mapping_confidence": mapping_conf,
                            "mapping_score": mapping_score,
                            "alt_deal": alt_deal,
                            "tier": tier,
                            "direction": direction,
                            "direction_conf": dir_conf,
                            "category": category,
                            "category_conf": cat_conf,
                            "impact": impact,
                            "impact_conf": impact_conf,
                            "impact_fallback": impact_fallback,
                            "severity_used": severity,
                            "engine_updated": engine_updated,
                            "headline": headline,
                        }
                    )
            except Exception as exc:
                locked_print(self.print_lock, f"  {self.colors.RED}News monitor error:{self.colors.RESET} {exc}")
            self.shutdown_event.wait(self.poll_interval)

    def summary(self):
        with self.log_lock:
            e = list(self.log)
        return {
            "processed": len(e),
            "signals": sum(1 for x in e if x["tier"] == "SIGNAL"),
            "category_only": sum(1 for x in e if x["tier"] == "CATEGORY"),
            "low": sum(1 for x in e if x["tier"] == "LOW"),
            "mapped": sum(1 for x in e if x["tier"] == "SIGNAL" and x["resolved_deal"]),
            "unmapped": sum(1 for x in e if x["tier"] == "SIGNAL" and not x["resolved_deal"]),
            "engine_updated": sum(1 for x in e if x["engine_updated"]),
        }

    def print_summary(self):
        s = self.summary()
        with self.print_lock:
            print("\n" + "=" * 78)
            print(f"  {self.colors.BOLD}NEWS SUMMARY{self.colors.RESET}")
            print("=" * 78)
            print(f"  Processed:      {s['processed']:>6}")
            print(f"  SIGNAL tier:    {s['signals']:>6}")
            print(f"  CATEGORY tier:  {s['category_only']:>6}")
            print(f"  LOW tier:       {s['low']:>6}")
            print(f"  Mapped signals: {s['mapped']:>6}")
            print(f"  Unmapped:       {s['unmapped']:>6}")
            print(f"  Engine updates: {s['engine_updated']:>6}")
            print("=" * 78 + "\n")

    def save_csv(self):
        with self.log_lock:
            e = list(self.log)
        if not e:
            return None
        filename = f"news_log_{datetime.now().strftime('%H%M%S')}.csv"
        fields = [
            "timestamp", "tick", "news_id", "ticker", "resolved_deal", "mapping_source",
            "mapping_confidence", "mapping_score", "alt_deal", "tier", "direction",
            "direction_conf", "category", "category_conf", "impact", "impact_conf",
            "impact_fallback", "severity_used", "engine_updated", "headline",
        ]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(e)
        return filename

def parse_trade_deals(tokens):
    deals = []
    for token in tokens:
        for part in re.split(r"[,\s]+", token):
            deal = part.strip().upper()
            if not deal:
                continue
            if deal == "ALL":
                for d in DEAL_IDS:
                    if d not in deals:
                        deals.append(d)
                continue
            if deal in DEALS:
                if deal not in deals:
                    deals.append(deal)
                continue
            return [], deal
    return deals, None


def run_command_loop(engine, shutdown_event, colors, print_lock):
    locked_print(
        print_lock,
        f"\n{colors.BOLD}Commands:{colors.RESET} {colors.GREEN}b/s d1 [d2 ...] qty{colors.RESET} | "
        f"{colors.GREEN}b/s all qty{colors.RESET} | "
        f"{colors.CYAN}status{colors.RESET} | {colors.YELLOW}rehedge{colors.RESET} | "
        f"{colors.RED}close{colors.RESET} | prob d1 0.75 | impl d1 | q\n",
    )
    while not shutdown_event.is_set():
        try:
            cmd = locked_input(print_lock, f"{colors.DIM}>{colors.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            shutdown_event.set()
            break
        if not cmd:
            continue
        p = cmd.split()
        h = p[0]
        if h in ("q", "quit", "exit"):
            shutdown_event.set()
        elif h in ("status", "st"):
            engine.show_status()
        elif h in ("rehedge", "rh", "hedge"):
            engine.rehedge_all()
        elif h in ("close", "flatten", "cl"):
            if locked_input(print_lock, f"  {colors.RED}Close ALL positions? [y/N]: {colors.RESET}").strip().lower() == "y":
                engine.close_all()
        elif h in ("b", "s"):
            side = "BUY" if h == "b" else "SELL"
            args = [x for x in p[1:] if x not in ("share", "shares", "sh")]
            if len(args) < 2:
                locked_print(print_lock, f"  {colors.RED}Usage: {h} d1 [d2 ...] 1000  |  {h} all 1000{colors.RESET}")
                continue
            try:
                qty = int(args[-1])
            except ValueError:
                locked_print(print_lock, f"  {colors.RED}Usage: {h} d1 [d2 ...] 1000  |  {h} all 1000{colors.RESET}")
                continue
            if qty <= 0:
                locked_print(print_lock, f"  {colors.RED}Quantity must be positive.{colors.RESET}")
                continue
            deals, bad_token = parse_trade_deals(args[:-1])
            if bad_token:
                locked_print(print_lock, f"  {colors.RED}Unknown deal token: {bad_token}{colors.RESET}")
                continue
            if not deals:
                locked_print(print_lock, f"  {colors.RED}No valid deals supplied.{colors.RESET}")
                continue
            locked_print(print_lock, f"\n  {colors.BOLD}Batch:{colors.RESET} {side} {qty} each across {', '.join(deals)}")
            for deal in deals:
                engine.execute_trade(deal, side, qty)
        elif h == "prob" and len(p) >= 3:
            deal = p[1].upper()
            if deal not in DEALS:
                locked_print(print_lock, f"  {colors.RED}Unknown deal: {deal}{colors.RESET}")
                continue
            try:
                new_p = float(p[2])
            except ValueError:
                locked_print(print_lock, f"  {colors.RED}Usage: prob d1 0.75{colors.RESET}")
                continue
            if not (0 <= new_p <= 1):
                locked_print(print_lock, f"  {colors.RED}Probability must be 0..1{colors.RESET}")
                continue
            old_p = engine.set_probability(deal, new_p)
            fair = engine.get_fair_value(deal, new_p)
            mkt = engine.get_market_price(DEALS[deal]["target"])
            edge = (fair - mkt) if mkt is not None else 0.0
            locked_print(print_lock, f"  {deal}: p {old_p:.1%} -> {new_p:.1%} Fair=${fair:.2f} Edge={(f'${edge:+.2f}' if mkt is not None else 'N/A')}")
        elif h in ("impl", "implied") and len(p) >= 2:
            deal = p[1].upper()
            if deal not in DEALS:
                locked_print(print_lock, f"  {colors.RED}Unknown deal: {deal}{colors.RESET}")
                continue
            p_impl = engine.implied_prob(deal)
            if p_impl is None:
                locked_print(print_lock, f"  {colors.RED}Cannot compute implied probability for {deal}.{colors.RESET}")
            else:
                p_model = engine.get_probability(deal)
                locked_print(print_lock, f"  {deal}: Market p={p_impl:.1%}  Model p={p_model:.1%}  Diff={p_model - p_impl:+.1%}")
        else:
            locked_print(print_lock, f"  {colors.DIM}Unknown command.{colors.RESET}")


def parse_args():
    p = argparse.ArgumentParser(description="Integrated merger arb runtime (news + trader)")
    p.add_argument("--model", default="merger_classifier_v3.joblib", help="Model file")
    p.add_argument("--api-key", default=API_KEY_DEFAULT, help="RIT API key")
    p.add_argument("--dir-threshold", type=float, default=0.70, help="Direction confidence threshold")
    p.add_argument("--cat-threshold", type=float, default=0.70, help="Category confidence threshold (display-only)")
    p.add_argument("--poll-interval", type=float, default=0.5, help="News poll interval")
    p.add_argument("--news-limit", type=int, default=20, help="News fetch limit")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return p.parse_args()


def install_signal_handler(shutdown_event, print_lock, colors):
    def _handler(_signum, _frame):
        shutdown_event.set()
        locked_print(print_lock, f"\n{colors.YELLOW}[SHUTDOWN] Ctrl+C received. Stopping...{colors.RESET}")
    signal.signal(signal.SIGINT, _handler)


def main():
    args = parse_args()
    if os.name == "nt" and not args.no_color:
        os.system("")
    colors = Colors(enabled=not args.no_color)
    print_lock = threading.RLock()
    shutdown_event = threading.Event()
    install_signal_handler(shutdown_event, print_lock, colors)

    session = requests.Session()
    session.headers.update({"X-API-Key": args.api_key})

    locked_print(print_lock, f"\n  Loading model: {args.model}")
    try:
        classifier = NewsClassifier(args.model, dir_threshold=args.dir_threshold, cat_threshold=args.cat_threshold)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1
    locked_print(
        print_lock,
        f"  High-confidence thresholds: Direction>={args.dir_threshold:.0%}  Category>={args.cat_threshold:.0%}",
    )

    engine = MergerArbEngine(session, colors, print_lock)
    locked_print(print_lock, f"\n{colors.YELLOW}Connecting to RIT at {BASE_URL}...{colors.RESET}")
    connected = False
    for i in range(10):
        if shutdown_event.is_set():
            break
        tick, status, name = get_case_status(session)
        if status != "DISCONNECTED":
            connected = True
            locked_print(print_lock, f"{colors.GREEN}Connected! Case: {name} | Status: {status} | Tick: {tick}{colors.RESET}")
            break
        locked_print(print_lock, f"  Retry {i + 1}/10...")
        time.sleep(1)
    if not connected:
        locked_print(print_lock, f"{colors.RED}Could not connect now. Running with retries in background.{colors.RESET}")

    engine.initialize()
    engine.show_status()

    monitor = NewsMonitor(
        classifier=classifier,
        engine=engine,
        session=session,
        dir_threshold=args.dir_threshold,
        cat_threshold=args.cat_threshold,
        poll_interval=args.poll_interval,
        news_limit=args.news_limit,
        colors=colors,
        print_lock=print_lock,
        shutdown_event=shutdown_event,
    )
    t = threading.Thread(target=monitor.run, name="news-monitor", daemon=True)
    t.start()

    run_command_loop(engine, shutdown_event, colors, print_lock)
    shutdown_event.set()
    t.join(timeout=3.0)

    monitor.print_summary()
    log_file = monitor.save_csv()
    if log_file:
        locked_print(print_lock, f"  Log saved: {log_file}")
    locked_print(print_lock, f"\n{colors.DIM}merger.py stopped.{colors.RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
