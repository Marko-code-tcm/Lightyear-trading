#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import smtplib
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# VERSION 2026-04-13-ALPHAVANTAGE

APP_DIR = Path(".")
STATE_DIR = APP_DIR / "portfolio_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_FILE = STATE_DIR / "portfolio.json"
SETTINGS_FILE = STATE_DIR / "settings.json"
HISTORY_FILE = STATE_DIR / "signal_history.csv"
TRANSACTIONS_FILE = STATE_DIR / "transactions.csv"

FIXED_NOTIFICATION_EMAIL = "marko.johanson@icloud.com"

DEFAULT_BENCHMARK = "SPY"
MIN_AVG_DOLLAR_VOLUME = 500_000.0
SELFTEST_SYMBOLS = ["SPY", "AAPL", "MSFT"]

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "JPM", "BAC", "GS", "XOM", "CVX",
    "LLY", "UNH", "COST", "CRM", "ORCL", "ADBE",
    "ASML", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MU", "INTU",
    "NOW", "PANW", "CRWD", "SHOP", "PLTR", "UBER", "ABNB", "INTC",
    "IBM", "CSCO", "ANET", "CDNS", "SNPS", "ADI", "NXPI",
    "V", "MA", "AXP", "BLK", "MS", "WFC", "C",
    "JNJ", "MRK", "ABBV", "PFE", "TMO", "DHR", "ISRG", "ABT",
    "AMGN", "REGN", "VRTX", "ZTS", "HCA", "IDXX",
    "GE", "HON", "CAT", "DE", "ETN", "PH", "TT", "EMR",
    "RTX", "LHX", "NOC", "GD", "UNP", "CSX", "NSC", "FDX", "UPS",
    "COP", "EOG", "SLB", "MPC", "VLO", "OXY", "LIN", "APD", "ECL",
    "FCX", "NEM", "NUE", "STLD", "FSLR",
    "NEE", "DUK", "SO", "AEP", "XEL",
    "WMT", "PG", "KO", "PEP", "MCD", "CMG", "NKE", "LULU",
    "TJX", "LOW", "HD", "TGT", "AZO", "ORLY", "BKNG",
    "TMUS", "VZ", "T", "CMCSA", "CHTR", "DIS", "SPOT",
    "EA", "TTWO",
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "DLR", "VICI",
    "LEN", "DHI", "NVR",
    "ON", "TER", "MPWR", "SWKS", "QRVO", "OLED", "GFS", "JBL", "FLEX",
    "GLW", "KEYS", "AKAM", "CYBR",
    "TOST", "DUOL", "AXON", "SOFI", "HOOD", "COIN", "AFRM", "UPST",
    "BILL", "ESTC", "TSM", "SAP", "NVO", "AZN", "SHEL", "BP",
    "RIO", "BHP", "RELX", "NGG", "HSBC", "UBS", "SAN", "ING", "DB",
    "SNY", "NVS", "BUD", "TTE", "PBR", "VALE", "ITUB", "CRH",
]


def default_settings() -> Dict:
    return {
        "investable_amount": 10000.0,
        "benchmark": DEFAULT_BENCHMARK,
        "user_universe": DEFAULT_UNIVERSE,
        "position_mode": "Equal weight",
        "max_new_positions_per_day": 3,
        "min_cash_buffer_pct": 0.05,
        "last_scan_date": None,
        "notifications_enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": "",
        "signal_strictness": 3,
        "alpha_vantage_api_key": "",
    }


def default_portfolio() -> Dict:
    return {
        "cash": 10000.0,
        "initial_capital": 10000.0,
        "positions": {},
        "closed_positions": [],
        "last_updated": None,
    }


def read_json(path: Path, fallback: Dict) -> Dict:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_settings() -> Dict:
    return default_settings() | read_json(SETTINGS_FILE, default_settings())


def save_settings(settings: Dict) -> None:
    write_json(SETTINGS_FILE, settings)


def load_portfolio() -> Dict:
    portfolio = default_portfolio() | read_json(PORTFOLIO_FILE, default_portfolio())
    portfolio.setdefault("positions", {})
    portfolio.setdefault("closed_positions", [])
    return portfolio


def save_portfolio(portfolio: Dict) -> None:
    portfolio["last_updated"] = datetime.now().isoformat()
    write_json(PORTFOLIO_FILE, portfolio)


def append_csv_row(path: Path, row: Dict) -> None:
    df_new = pd.DataFrame([row])
    if path.exists():
        try:
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(path, index=False)


def unique_symbols(symbols: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in symbols:
        s2 = str(s).strip().upper()
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out


def get_thresholds(strictness: int) -> Tuple[float, float]:
    mapping = {
        1: (2.5, -2.0),
        2: (3.5, -3.0),
        3: (4.5, -4.0),
        4: (5.5, -5.0),
    }
    return mapping.get(strictness, (4.5, -4.0))


def get_strictness_label(strictness: int) -> str:
    labels = {
        1: "Leebe",
        2: "Tavaline",
        3: "Range",
        4: "Väga range",
    }
    return labels.get(strictness, "Range")


def alpha_vantage_url(symbol: str, api_key: str) -> str:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "datatype": "json",
        "apikey": api_key,
    }
    return "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)


def download_symbol_history(symbol: str, api_key: str, timeout: int = 20) -> pd.DataFrame:
    if not api_key.strip():
        return pd.DataFrame()

    url = alpha_vantage_url(symbol, api_key)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception:
        return pd.DataFrame()

    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        return pd.DataFrame()

    rows = []
    for dt, values in series.items():
        try:
            rows.append(
                {
                    "Date": pd.to_datetime(dt),
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": float(values["5. volume"]),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df.dropna()


def download_universe_data(symbols: List[str], api_key: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for symbol in unique_symbols(symbols):
        df = download_symbol_history(symbol, api_key=api_key)
        if not df.empty:
            out[symbol] = df
    return out


def run_data_source_selftest(api_key: str) -> Dict:
    loaded = {}
    failures = []

    for symbol in SELFTEST_SYMBOLS:
        df = download_symbol_history(symbol, api_key=api_key)
        if df.empty:
            failures.append(symbol)
        else:
            loaded[symbol] = len(df)

    return {
        "tested": SELFTEST_SYMBOLS,
        "loaded_count": len(loaded),
        "loaded_rows": loaded,
        "failures": failures,
        "ok": len(loaded) > 0,
    }


def compute_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=lookback, min_periods=lookback).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["AvgVol20"] = x["Volume"].rolling(20).mean()
    x["DollarVol20"] = x["Close"] * x["AvgVol20"]
    x["ATR14"] = compute_atr(x, 14)
    x["High20"] = x["High"].rolling(20).max().shift(1)
    x["Low20"] = x["Low"].rolling(20).min().shift(1)
    x["Mom63"] = x["Close"] / x["Close"].shift(63) - 1.0
    return x


def compute_market_regime(benchmark_df: Optional[pd.DataFrame]) -> str:
    if benchmark_df is None or benchmark_df.empty:
        return "NEUTRAL"

    x = benchmark_df.copy()
    x["SMA20"] = x["Close"].rolling(20).mean()
    x["SMA50"] = x["Close"].rolling(50).mean()
    latest = x.iloc[-1]

    if pd.isna(latest["SMA20"]) or pd.isna(latest["SMA50"]):
        return "NEUTRAL"
    if latest["Close"] > latest["SMA50"] and latest["SMA20"] > latest["SMA50"]:
        return "BULL"
    if latest["Close"] < latest["SMA50"] and latest["SMA20"] < latest["SMA50"]:
        return "BEAR"
    return "NEUTRAL"


def relative_strength_score(symbol_df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame], lookback: int = 40) -> float:
    if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
        return np.nan

    joined = pd.concat(
        [
            symbol_df[["Close"]].rename(columns={"Close": "sym_close"}),
            benchmark_df[["Close"]].rename(columns={"Close": "bench_close"}),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(joined) < lookback + 5:
        return np.nan

    rs = joined["sym_close"] / joined["bench_close"]
    return float(rs.iloc[-1] / rs.iloc[-lookback] - 1.0)


def score_symbol(
    symbol: str,
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    regime: str,
    strictness: int,
) -> Dict:
    x = add_indicators(df)

    if len(x) < 70:
        return {"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "not enough history"}

    latest = x.iloc[-1]
    prev = x.iloc[-2] if len(x) >= 2 else latest

    price = float(latest["Close"])
    avg_dollar_vol = float(latest["DollarVol20"]) if pd.notna(latest["DollarVol20"]) else np.nan

    if price < 5:
        return {"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "price too low"}
    if np.isnan(avg_dollar_vol) or avg_dollar_vol < MIN_AVG_DOLLAR_VOLUME:
        return {"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "insufficient liquidity"}

    rs = relative_strength_score(x, benchmark_df)
    atr = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else np.nan
    atr_pct = float(atr / price) if price > 0 and pd.notna(atr) else np.nan
    mom63 = float(latest["Mom63"]) if pd.notna(latest["Mom63"]) else np.nan

    score = 0.0
    reasons: List[str] = []

    if pd.notna(latest["SMA20"]) and price > latest["SMA20"]:
        score += 1.0
        reasons.append("above SMA20")
    else:
        score -= 1.0

    if pd.notna(latest["SMA50"]) and price > latest["SMA50"]:
        score += 2.0
        reasons.append("above SMA50")
    else:
        score -= 2.0

    if pd.notna(mom63):
        if mom63 > 0.12:
            score += 2.0
            reasons.append("strong momentum")
        elif mom63 > 0.04:
            score += 1.0
        elif mom63 < -0.08:
            score -= 2.0
        elif mom63 < 0:
            score -= 1.0

    if pd.notna(rs):
        if rs > 0.08:
            score += 1.5
            reasons.append("relative strength")
        elif rs < -0.05:
            score -= 1.5

    if pd.notna(latest["High20"]) and price > latest["High20"]:
        score += 1.5
        reasons.append("20d breakout")
    if pd.notna(latest["Low20"]) and price < latest["Low20"]:
        score -= 1.5

    if pd.notna(latest["AvgVol20"]) and latest["AvgVol20"] > 0:
        vol_ratio = float(latest["Volume"] / latest["AvgVol20"])
        if vol_ratio > 1.2 and price > float(prev["Close"]):
            score += 1.0
        elif vol_ratio > 1.2 and price < float(prev["Close"]):
            score -= 1.0
    else:
        vol_ratio = np.nan

    if pd.notna(atr_pct):
        if atr_pct > 0.10:
            score -= 1.0
        elif atr_pct < 0.04:
            score += 0.5

    if regime == "BULL" and score > 0:
        score += 0.5
    elif regime == "BEAR":
        score -= 1.0

    buy_threshold, sell_threshold = get_thresholds(strictness)

    action = "HOLD"
    if score >= buy_threshold and regime != "BEAR":
        action = "BUY"
    elif score <= sell_threshold:
        action = "SELL"

    confidence = "Low"
    if strictness == 1:
        if abs(score) >= 4.5:
            confidence = "High"
        elif abs(score) >= 3.5:
            confidence = "Medium"
    elif strictness == 2:
        if abs(score) >= 5.0:
            confidence = "High"
        elif abs(score) >= 4.0:
            confidence = "Medium"
    elif strictness == 3:
        if abs(score) >= 5.5:
            confidence = "High"
        elif abs(score) >= 4.5:
            confidence = "Medium"
    else:
        if abs(score) >= 6.0:
            confidence = "High"
        elif abs(score) >= 5.0:
            confidence = "Medium"

    stop_ref = np.nan
    if pd.notna(atr):
        stop_ref = price - 2.5 * atr if action == "BUY" else price + 2.5 * atr

    return {
        "symbol": symbol,
        "action": action,
        "score": round(float(score), 3),
        "signal_price": round(price, 4),
        "close": round(price, 4),
        "mom63": round(mom63, 4) if pd.notna(mom63) else np.nan,
        "rs_vs_benchmark": round(rs, 4) if pd.notna(rs) else np.nan,
        "atr_pct": round(atr_pct, 4) if pd.notna(atr_pct) else np.nan,
        "volume_ratio": round(vol_ratio, 4) if pd.notna(vol_ratio) else np.nan,
        "avg_dollar_vol_20": round(avg_dollar_vol, 2),
        "stop_reference": round(float(stop_ref), 4) if pd.notna(stop_ref) else np.nan,
        "confidence": confidence,
        "conviction": get_strictness_label(strictness),
        "reason": "; ".join(reasons[:8]),
    }


def run_daily_scan(
    universe: List[str],
    benchmark: str,
    strictness: int,
    api_key: str,
) -> Tuple[str, pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    universe = unique_symbols(universe)
    symbols = unique_symbols(universe + [benchmark])
    data = download_universe_data(symbols, api_key=api_key)

    benchmark_df = data.get(benchmark)
    regime = compute_market_regime(benchmark_df)

    missing_symbols = [s for s in universe if s not in data]

    rows: List[Dict] = []
    for symbol in universe:
        df = data.get(symbol)
        if df is None or df.empty:
            rows.append({"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "missing data"})
            continue

        try:
            result = score_symbol(symbol, df, benchmark_df, regime, strictness)
            if benchmark_df is None:
                extra = f"benchmark unavailable: {benchmark}"
                result["reason"] = f"{result.get('reason', '')}; {extra}".strip("; ")
            rows.append(result)
        except Exception as exc:
            rows.append({"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": f"error: {exc}"})

    signals = pd.DataFrame(rows)
    if not signals.empty:
        signals["scan_date"] = pd.Timestamp(date.today())
        signals = signals.sort_values(["action", "score"], ascending=[True, False])

    return regime, signals, data, missing_symbols


def summarize_signals(signals: pd.DataFrame) -> Dict[str, int]:
    if signals.empty or "action" not in signals.columns:
        return {"BUY": 0, "HOLD": 0, "SELL": 0, "SKIP": 0}
    counts = signals["action"].value_counts().to_dict()
    return {
        "BUY": int(counts.get("BUY", 0)),
        "HOLD": int(counts.get("HOLD", 0)),
        "SELL": int(counts.get("SELL", 0)),
        "SKIP": int(counts.get("SKIP", 0)),
    }


def summarize_skip_reasons(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "action" not in signals.columns or "reason" not in signals.columns:
        return pd.DataFrame()

    skip_df = signals[signals["action"] == "SKIP"].copy()
    if skip_df.empty:
        return pd.DataFrame()

    reasons = skip_df["reason"].fillna("unknown").astype(str).str.strip()
    counts = Counter(reasons)
    out = pd.DataFrame({"reason": list(counts.keys()), "count": list(counts.values())})
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def top_signals_df(signals: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    df = signals.copy()
    df = df[df["action"] != "SKIP"].copy() if "action" in df.columns else df
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("score", ascending=False).head(n)
    cols = ["symbol", "action", "score", "confidence", "conviction", "signal_price", "close", "reason"]
    return df[[c for c in cols if c in df.columns]].copy()


def normalize_positions(portfolio: Dict) -> None:
    for symbol, pos in portfolio["positions"].items():
        pos.setdefault("symbol", symbol)
        pos.setdefault("quantity", 0)
        pos.setdefault("avg_price", 0.0)
        pos.setdefault("market_value", 0.0)
        pos.setdefault("last_price", 0.0)
        pos.setdefault("status", "HOLD")
        pos.setdefault("stop_reference", None)
        pos.setdefault("opened_at", datetime.now().isoformat())
        pos.setdefault("last_signal_score", None)
        pos.setdefault("last_signal_reason", "")


def estimate_portfolio_value(portfolio: Dict) -> float:
    total = float(portfolio.get("cash", 0.0))
    for pos in portfolio.get("positions", {}).values():
        total += float(pos.get("market_value", 0.0))
    return total


def update_position_marks(portfolio: Dict, signals: pd.DataFrame) -> None:
    normalize_positions(portfolio)
    signal_map = {row["symbol"]: row for _, row in signals.iterrows()}

    for symbol, pos in portfolio["positions"].items():
        row = signal_map.get(symbol)
        if row is None:
            continue
        last_price = float(row.get("close", pos.get("last_price", 0.0)))
        qty = float(pos.get("quantity", 0.0))
        pos["last_price"] = last_price
        pos["market_value"] = round(last_price * qty, 2)
        pos["status"] = "SELL" if row.get("action") == "SELL" else "HOLD"
        pos["last_signal_score"] = None if pd.isna(row.get("score")) else float(row.get("score"))
        pos["last_signal_reason"] = str(row.get("reason", ""))
        pos["stop_reference"] = None if pd.isna(row.get("stop_reference")) else float(row.get("stop_reference"))


def current_open_symbols(portfolio: Dict) -> set:
    return set(portfolio.get("positions", {}).keys())


def available_cash_for_new_positions(portfolio: Dict, settings: Dict) -> float:
    cash = float(portfolio.get("cash", 0.0))
    min_buffer = estimate_portfolio_value(portfolio) * float(settings.get("min_cash_buffer_pct", 0.05))
    return max(0.0, cash - min_buffer)


def determine_buy_budget(portfolio: Dict, settings: Dict, num_candidates: int) -> float:
    if num_candidates <= 0:
        return 0.0
    free_cash = available_cash_for_new_positions(portfolio, settings)
    if free_cash <= 0:
        return 0.0

    slots = max(1, min(num_candidates, int(settings.get("max_new_positions_per_day", 3))))
    if settings.get("position_mode", "Equal weight") == "Equal weight":
        current_positions = len(portfolio.get("positions", {}))
        target_positions = max(current_positions + slots, 1)
        target_size = estimate_portfolio_value(portfolio) / target_positions
        return max(0.0, min(free_cash / slots, target_size))

    return free_cash / slots


def build_actionable_proposals(portfolio: Dict, settings: Dict, signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    open_symbols = current_open_symbols(portfolio)
    rows = []

    buys = signals[(signals["action"] == "BUY") & (~signals["symbol"].isin(open_symbols))].copy()
    buys = buys.sort_values("score", ascending=False)
    budget_per_buy = determine_buy_budget(portfolio, settings, len(buys))

    for _, row in buys.iterrows():
        price = float(row["close"])
        qty = int(math.floor(budget_per_buy / price)) if price > 0 else 0
        est_cost = round(qty * price, 2)
        if qty <= 0:
            continue
        rows.append({
            **row.to_dict(),
            "proposal_type": "BUY",
            "quantity": qty,
            "estimated_cash_impact": -est_cost,
            "note": "Open new position",
        })

    for _, row in signals.iterrows():
        symbol = str(row["symbol"])
        if symbol not in open_symbols:
            continue
        pos = portfolio["positions"][symbol]
        qty = int(pos.get("quantity", 0))
        price = float(row["close"])
        proposal = "SELL" if row["action"] == "SELL" else "HOLD"
        rows.append({
            **row.to_dict(),
            "proposal_type": proposal,
            "quantity": qty,
            "estimated_cash_impact": round(qty * price, 2) if proposal == "SELL" else 0.0,
            "note": "Close position" if proposal == "SELL" else "Keep holding",
        })

    proposals = pd.DataFrame(rows)
    if proposals.empty:
        return proposals

    order_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
    proposals["order_key"] = proposals["proposal_type"].map(order_map)
    return proposals.sort_values(["order_key", "score"], ascending=[True, False]).drop(columns=["order_key"])


def accept_buy(portfolio: Dict, symbol: str, quantity: int, price: float, reason: str, score: Optional[float], stop_reference: Optional[float]) -> str:
    cost = round(quantity * price, 2)
    cash = float(portfolio.get("cash", 0.0))
    if quantity <= 0:
        return "Kogus peab olema suurem kui 0."
    if cost > cash:
        return f"Vaba raha ei piisa. Vajalik {cost:.2f}, olemas {cash:.2f}."
    if symbol in portfolio.get("positions", {}):
        return f"{symbol} on juba portfellis."

    portfolio["cash"] = round(cash - cost, 2)
    portfolio["positions"][symbol] = {
        "symbol": symbol,
        "quantity": int(quantity),
        "avg_price": round(price, 4),
        "last_price": round(price, 4),
        "market_value": round(cost, 2),
        "status": "HOLD",
        "stop_reference": None if stop_reference is None or pd.isna(stop_reference) else round(float(stop_reference), 4),
        "opened_at": datetime.now().isoformat(),
        "last_signal_score": None if score is None or pd.isna(score) else float(score),
        "last_signal_reason": reason,
    }
    append_csv_row(
        TRANSACTIONS_FILE,
        {
            "timestamp": datetime.now().isoformat(),
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": round(price, 4),
            "gross_amount": cost,
            "reason": reason,
        },
    )
    save_portfolio(portfolio)
    return f"Ost aktsepteeritud: {symbol}, {quantity} tk hinnaga {price:.2f}."


def accept_sell(portfolio: Dict, symbol: str, price: float, reason: str, score: Optional[float]) -> str:
    pos = portfolio.get("positions", {}).get(symbol)
    if not pos:
        return f"{symbol} ei ole portfellis."

    qty = int(pos.get("quantity", 0))
    if qty <= 0:
        return f"{symbol} kogus on vigane."

    proceeds = round(qty * price, 2)
    avg_price = float(pos.get("avg_price", 0.0))
    pnl = round((price - avg_price) * qty, 2)

    portfolio["cash"] = round(float(portfolio.get("cash", 0.0)) + proceeds, 2)
    portfolio["closed_positions"].append({
        **pos,
        "closed_at": datetime.now().isoformat(),
        "exit_price": round(price, 4),
        "realized_pnl": pnl,
        "close_reason": reason,
    })
    del portfolio["positions"][symbol]

    append_csv_row(
        TRANSACTIONS_FILE,
        {
            "timestamp": datetime.now().isoformat(),
            "type": "SELL",
            "symbol": symbol,
            "quantity": qty,
            "price": round(price, 4),
            "gross_amount": proceeds,
            "reason": reason,
            "realized_pnl": pnl,
        },
    )
    save_portfolio(portfolio)
    return f"Müük aktsepteeritud: {symbol}, {qty} tk hinnaga {price:.2f}, P/L {pnl:.2f}."


def send_email_message(host: str, port: int, username: str, password: str, to_addr: str, text: str) -> Tuple[bool, str]:
    if not all([host, port, username, password, to_addr]):
        return False, "SMTP settings are incomplete"
    try:
        msg = MIMEText(text, "plain", "utf-8")
        msg["Subject"] = f"Daily portfolio scan {date.today().isoformat()}"
        msg["From"] = username
        msg["To"] = to_addr
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email error: {exc}"


def build_notification_message(regime: str, proposals: pd.DataFrame, settings: Dict) -> str:
    lines = [
        f"Daily portfolio scan — {date.today().isoformat()}",
        f"Benchmark: {settings.get('benchmark')} | Regime: {regime}",
        f"Strictness: {get_strictness_label(int(settings.get('signal_strictness', 3)))}",
        "",
    ]
    for label in ["BUY", "SELL", "HOLD"]:
        subset = proposals[proposals["proposal_type"] == label]
        lines.append(label)
        if subset.empty:
            lines.append("- none")
        else:
            for _, row in subset.head(5).iterrows():
                qty_text = f" | qty {int(row['quantity'])}" if pd.notna(row.get("quantity")) else ""
                lines.append(f"- {row['symbol']} @ {float(row['close']):.2f} | score {float(row['score']):+.2f}{qty_text}")
        lines.append("")
    return "\n".join(lines).strip()


def send_notifications(settings: Dict, regime: str, proposals: pd.DataFrame) -> Tuple[bool, str]:
    if not settings.get("notifications_enabled", False):
        return False, "Notifications are disabled"
    text = build_notification_message(regime, proposals, settings)
    return send_email_message(
        settings.get("smtp_host", "smtp.gmail.com"),
        int(settings.get("smtp_port", 587)),
        settings.get("smtp_username", ""),
        settings.get("smtp_password", ""),
        FIXED_NOTIFICATION_EMAIL,
        text,
    )


def execute_daily_job(settings: Dict, portfolio: Dict) -> Dict:
    api_key = str(settings.get("alpha_vantage_api_key", "")).strip()
    selftest = run_data_source_selftest(api_key)
    if not selftest["ok"]:
        return {
            "regime": "UNKNOWN",
            "signals": pd.DataFrame(),
            "proposals": pd.DataFrame(),
            "notification_ok": False,
            "notification_message": "Data source self-test failed",
            "data_keys": [],
            "missing_symbols": [],
            "selftest": selftest,
        }

    universe = unique_symbols([str(x).strip().upper() for x in settings.get("user_universe", []) if str(x).strip()])
    benchmark = str(settings.get("benchmark", DEFAULT_BENCHMARK)).upper().strip()
    strictness = int(settings.get("signal_strictness", 3))

    if benchmark not in universe:
        universe.append(benchmark)

    regime, signals, data, missing_symbols = run_daily_scan(
        universe=universe,
        benchmark=benchmark,
        strictness=strictness,
        api_key=api_key,
    )

    settings["last_scan_date"] = str(date.today())
    save_settings(settings)

    update_position_marks(portfolio, signals)
    save_portfolio(portfolio)

    proposals = build_actionable_proposals(portfolio, settings, signals)

    if not signals.empty:
        export = signals.copy()
        export["benchmark"] = benchmark
        export["regime"] = regime
        export["strictness"] = strictness
        export.to_csv(HISTORY_FILE, index=False)

    notif_ok, notif_msg = send_notifications(settings, regime, proposals)

    return {
        "regime": regime,
        "signals": signals,
        "proposals": proposals,
        "notification_ok": notif_ok,
        "notification_message": notif_msg,
        "data_keys": sorted(list(data.keys())),
        "missing_symbols": missing_symbols,
        "selftest": selftest,
    }


def fmt_money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


def portfolio_snapshot_df(portfolio: Dict) -> pd.DataFrame:
    rows = []
    for symbol, pos in portfolio.get("positions", {}).items():
        qty = float(pos.get("quantity", 0.0))
        avg_price = float(pos.get("avg_price", 0.0))
        last_price = float(pos.get("last_price", 0.0))
        mv = float(pos.get("market_value", qty * last_price))
        rows.append({
            "symbol": symbol,
            "quantity": int(qty),
            "avg_price": round(avg_price, 4),
            "last_price": round(last_price, 4),
            "market_value": round(mv, 2),
            "unrealized_pnl": round((last_price - avg_price) * qty, 2),
            "status": pos.get("status", "HOLD"),
            "stop_reference": pos.get("stop_reference"),
            "reason": pos.get("last_signal_reason", ""),
        })
    if not rows:
        return pd.DataFrame(columns=["symbol", "quantity", "avg_price", "last_price", "market_value", "unrealized_pnl", "status", "stop_reference", "reason"])
    return pd.DataFrame(rows)


def proposal_table_df(proposals: pd.DataFrame) -> pd.DataFrame:
    if proposals.empty:
        return proposals
    cols = ["symbol", "proposal_type", "score", "confidence", "conviction", "signal_price", "close", "quantity", "estimated_cash_impact", "stop_reference", "reason"]
    return proposals[[c for c in cols if c in proposals.columns]].copy()


def is_probably_mobile() -> bool:
    ua = st.query_params.get("ua", "")
    if isinstance(ua, list):
        ua = " ".join(ua)
    ua = str(ua).lower()
    return any(marker in ua for marker in ["iphone", "android", "mobile"])


def init_view_mode() -> None:
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "mobile" if is_probably_mobile() else "desktop"


def toggle_view_mode() -> None:
    st.session_state["view_mode"] = "desktop" if st.session_state["view_mode"] == "mobile" else "mobile"


def render_top_bar() -> None:
    left, right = st.columns([5, 1])
    with left:
        st.title("Portfolio Signal Dashboard V2")
    with right:
        label = "Desktop vaade" if st.session_state["view_mode"] == "mobile" else "Mobiilivaade"
        if st.button(label, use_container_width=True):
            toggle_view_mode()
            st.rerun()


def render_desktop_sidebar(settings: Dict) -> Tuple[float, str, str, str, int, float, bool, str, int, str, str, int, str]:
    with st.sidebar:
        st.header("Seaded")

        investable_amount = st.number_input("Investeeritav summa", min_value=0.0, value=float(settings.get("investable_amount", 10000.0)), step=100.0)
        benchmark = st.text_input("Benchmark", value=str(settings.get("benchmark", DEFAULT_BENCHMARK))).upper().strip()
        universe_text = st.text_area("Jälgitavad tickerid (komaga eraldatud)", value=", ".join(settings.get("user_universe", DEFAULT_UNIVERSE)), height=140)
        position_mode = st.selectbox("Uute ostude jaotus", options=["Equal weight", "Use available cash evenly"], index=0 if settings.get("position_mode", "Equal weight") == "Equal weight" else 1)
        max_new_positions_per_day = st.number_input("Maksimaalne uute positsioonide arv päevas", min_value=1, max_value=20, value=int(settings.get("max_new_positions_per_day", 3)), step=1)
        min_cash_buffer_pct = st.slider("Raha puhver osakaaluna portfellist", min_value=0.0, max_value=0.30, value=float(settings.get("min_cash_buffer_pct", 0.05)), step=0.01)

        signal_strictness = st.slider("Analüüsi rangus", min_value=1, max_value=4, value=int(settings.get("signal_strictness", 3)), step=1)
        st.caption(f"{get_strictness_label(signal_strictness)} — leebe annab rohkem signaale, range vähem aga tugevamaid.")

        st.subheader("Andmeallikas")
        api_key = st.text_input("Alpha Vantage API key", value=str(settings.get("alpha_vantage_api_key", "")), type="password")

        st.subheader("E-maili teavitus")
        notifications_enabled = st.checkbox("Teavitused lubatud", value=bool(settings.get("notifications_enabled", False)))
        st.caption(f"Teavitused lähevad aadressile: {FIXED_NOTIFICATION_EMAIL}")
        smtp_host = st.text_input("SMTP host", value=str(settings.get("smtp_host", "smtp.gmail.com")))
        smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(settings.get("smtp_port", 587)), step=1)
        smtp_username = st.text_input("SMTP username", value=str(settings.get("smtp_username", "")))
        smtp_password = st.text_input("SMTP password", value=str(settings.get("smtp_password", "")), type="password")

    return (
        investable_amount,
        benchmark,
        universe_text,
        position_mode,
        int(max_new_positions_per_day),
        float(min_cash_buffer_pct),
        bool(notifications_enabled),
        smtp_host,
        int(smtp_port),
        smtp_username,
        smtp_password,
        int(signal_strictness),
        api_key,
    )


def render_mobile_header(settings: Dict) -> Tuple[float, str, str, str, int, float, bool, str, int, str, str, int, str]:
    investable_amount = float(settings.get("investable_amount", 10000.0))
    benchmark = str(settings.get("benchmark", DEFAULT_BENCHMARK)).upper().strip()
    universe_text = ", ".join(settings.get("user_universe", DEFAULT_UNIVERSE))
    position_mode = settings.get("position_mode", "Equal weight")
    max_new_positions_per_day = int(settings.get("max_new_positions_per_day", 3))
    min_cash_buffer_pct = float(settings.get("min_cash_buffer_pct", 0.05))
    notifications_enabled = bool(settings.get("notifications_enabled", False))
    smtp_host = str(settings.get("smtp_host", "smtp.gmail.com"))
    smtp_port = int(settings.get("smtp_port", 587))
    smtp_username = str(settings.get("smtp_username", ""))
    smtp_password = str(settings.get("smtp_password", ""))
    signal_strictness = int(settings.get("signal_strictness", 3))
    api_key = str(settings.get("alpha_vantage_api_key", ""))

    st.caption("Mobiilivaade: ainult scan nupp, portfelli seis, top signaalid ja scan'i järel soovitused.")
    st.caption(f"Analüüsi rangus: {get_strictness_label(signal_strictness)}")
    return (
        investable_amount,
        benchmark,
        universe_text,
        position_mode,
        max_new_positions_per_day,
        min_cash_buffer_pct,
        notifications_enabled,
        smtp_host,
        smtp_port,
        smtp_username,
        smtp_password,
        signal_strictness,
        api_key,
    )


def save_ui_settings(
    settings: Dict,
    investable_amount: float,
    benchmark: str,
    universe_text: str,
    position_mode: str,
    max_new_positions_per_day: int,
    min_cash_buffer_pct: float,
    notifications_enabled: bool,
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    signal_strictness: int,
    api_key: str,
) -> Dict:
    parsed_universe = unique_symbols([x.strip().upper() for x in universe_text.split(",") if x.strip()])
    settings.update({
        "investable_amount": float(investable_amount),
        "benchmark": benchmark,
        "user_universe": parsed_universe,
        "position_mode": position_mode,
        "max_new_positions_per_day": int(max_new_positions_per_day),
        "min_cash_buffer_pct": float(min_cash_buffer_pct),
        "notifications_enabled": bool(notifications_enabled),
        "smtp_host": smtp_host,
        "smtp_port": int(smtp_port),
        "smtp_username": smtp_username,
        "smtp_password": smtp_password,
        "signal_strictness": int(signal_strictness),
        "alpha_vantage_api_key": api_key.strip(),
    })
    save_settings(settings)
    return settings


def render_portfolio_overview(portfolio: Dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    pv = estimate_portfolio_value(portfolio)
    cash = float(portfolio.get("cash", 0.0))
    c1.metric("Portfelli väärtus", fmt_money(pv))
    c2.metric("Vaba raha", fmt_money(cash))
    c3.metric("Investeeritud", fmt_money(pv - cash))
    c4.metric("Avatud positsioone", str(len(portfolio.get("positions", {}))))


def render_selftest(selftest: Dict) -> None:
    st.subheader("Andmeallika enesetest")
    c1, c2 = st.columns(2)
    c1.metric("Testitud sümbolid", len(selftest.get("tested", [])))
    c2.metric("Töötavad sümbolid", int(selftest.get("loaded_count", 0)))

    if selftest.get("loaded_rows"):
        df = pd.DataFrame([{"symbol": k, "rows": v} for k, v in selftest["loaded_rows"].items()]).sort_values("symbol")
        st.dataframe(df, use_container_width=True, hide_index=True)

    if selftest.get("failures"):
        st.warning("Need test-sümbolid ei tulnud sisse: " + ", ".join(selftest["failures"]))


def render_signal_summary(signals: pd.DataFrame) -> None:
    st.subheader("Signaalide kokkuvõte")
    counts = summarize_signals(signals)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BUY", counts["BUY"])
    c2.metric("HOLD", counts["HOLD"])
    c3.metric("SELL", counts["SELL"])
    c4.metric("SKIP", counts["SKIP"])


def render_skip_reasons(signals: pd.DataFrame) -> None:
    st.subheader("SKIP põhjused")
    skip_df = summarize_skip_reasons(signals)
    if skip_df.empty:
        st.write("SKIP põhjuseid ei ole.")
        return
    st.dataframe(skip_df.head(10), use_container_width=True, hide_index=True)


def render_top_signals(signals: pd.DataFrame) -> None:
    st.subheader("Top 10 signaalid täna")
    top_df = top_signals_df(signals, 10)
    if top_df.empty:
        st.write("Täna top signaale ei ole.")
        return
    st.dataframe(top_df, use_container_width=True, hide_index=True)


def render_loaded_missing(data_keys: List[str], missing_symbols: List[str]) -> None:
    st.subheader("Andmete diagnostika")
    c1, c2 = st.columns(2)
    c1.metric("Laetud tickerid", len(data_keys))
    c2.metric("Puuduvad tickerid", len(missing_symbols))

    if data_keys:
        with st.expander("Laetud tickerid"):
            st.write(", ".join(data_keys))
    if missing_symbols:
        with st.expander("Puuduvad tickerid"):
            st.write(", ".join(missing_symbols))


def render_proposals(proposals: pd.DataFrame, portfolio: Dict) -> None:
    st.subheader("Soovitused")

    if proposals.empty:
        st.write("Täna ettepanekuid veel ei ole.")
        return

    st.dataframe(proposal_table_df(proposals), use_container_width=True, hide_index=True)

    buy_df = proposals[proposals["proposal_type"] == "BUY"].sort_values("score", ascending=False)
    sell_df = proposals[proposals["proposal_type"] == "SELL"].sort_values("score", ascending=True)

    if not buy_df.empty:
        st.markdown("**BUY**")
        for _, row in buy_df.iterrows():
            sym = str(row["symbol"])
            with st.container(border=True):
                st.write(f"{sym} | score {row['score']} | signal price {row.get('signal_price', row['close'])} | hind {row['close']}")
                st.write(f"Kogus: {int(row['quantity'])} | Kulu: {fmt_money(abs(float(row['estimated_cash_impact'])))}")
                st.write(f"Põhjus: {row['reason']}")
                if st.button(f"Aksepteeri BUY {sym}", key=f"buy_{sym}", use_container_width=True):
                    msg = accept_buy(
                        portfolio,
                        sym,
                        int(row["quantity"]),
                        float(row["close"]),
                        str(row.get("reason", "")),
                        None if pd.isna(row.get("score")) else float(row.get("score")),
                        None if pd.isna(row.get("stop_reference")) else float(row.get("stop_reference")),
                    )
                    st.success(msg)
                    st.rerun()

    if not sell_df.empty:
        st.markdown("**SELL**")
        for _, row in sell_df.iterrows():
            sym = str(row["symbol"])
            with st.container(border=True):
                st.write(f"{sym} | score {row['score']} | signal price {row.get('signal_price', row['close'])} | hind {row['close']}")
                st.write(f"Kogus: {int(row['quantity'])} | Laekumine: {fmt_money(float(row['estimated_cash_impact']))}")
                st.write(f"Põhjus: {row['reason']}")
                if st.button(f"Aksepteeri SELL {sym}", key=f"sell_{sym}", use_container_width=True):
                    msg = accept_sell(
                        portfolio,
                        sym,
                        float(row["close"]),
                        str(row.get("reason", "")),
                        None if pd.isna(row.get("score")) else float(row.get("score")),
                    )
                    st.success(msg)
                    st.rerun()


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Portfolio Signal Dashboard V2", layout="wide")
    init_view_mode()
    render_top_bar()

    settings = load_settings()
    portfolio = load_portfolio()
    normalize_positions(portfolio)

    if st.session_state["view_mode"] == "desktop":
        (
            investable_amount,
            benchmark,
            universe_text,
            position_mode,
            max_new_positions_per_day,
            min_cash_buffer_pct,
            notifications_enabled,
            smtp_host,
            smtp_port,
            smtp_username,
            smtp_password,
            signal_strictness,
            api_key,
        ) = render_desktop_sidebar(settings)
    else:
        (
            investable_amount,
            benchmark,
            universe_text,
            position_mode,
            max_new_positions_per_day,
            min_cash_buffer_pct,
            notifications_enabled,
            smtp_host,
            smtp_port,
            smtp_username,
            smtp_password,
            signal_strictness,
            api_key,
        ) = render_mobile_header(settings)

    render_portfolio_overview(portfolio)
    st.write("")

    if st.session_state["view_mode"] == "desktop":
        left, _ = st.columns([1, 1])
        with left:
            if st.button("Salvesta seaded", use_container_width=True):
                save_ui_settings(
                    settings,
                    investable_amount,
                    benchmark,
                    universe_text,
                    position_mode,
                    max_new_positions_per_day,
                    min_cash_buffer_pct,
                    notifications_enabled,
                    smtp_host,
                    smtp_port,
                    smtp_username,
                    smtp_password,
                    signal_strictness,
                    api_key,
                )
                if not portfolio.get("positions") and not portfolio.get("closed_positions"):
                    portfolio["cash"] = float(investable_amount)
                    portfolio["initial_capital"] = float(investable_amount)
                    save_portfolio(portfolio)
                st.success("Seaded salvestatud.")

    run_scan_now = st.button("Run daily scan", type="primary", use_container_width=True)
    signals = pd.DataFrame()
    proposals = pd.DataFrame()
    data_keys: List[str] = []
    missing_symbols: List[str] = []
    selftest = run_data_source_selftest(api_key)

    if run_scan_now:
        try:
            settings = save_ui_settings(
                settings,
                investable_amount,
                benchmark,
                universe_text,
                position_mode,
                max_new_positions_per_day,
                min_cash_buffer_pct,
                notifications_enabled,
                smtp_host,
                smtp_port,
                smtp_username,
                smtp_password,
                signal_strictness,
                api_key,
            )

            if not portfolio.get("positions") and not portfolio.get("closed_positions"):
                portfolio["cash"] = float(investable_amount)
                portfolio["initial_capital"] = float(investable_amount)
                save_portfolio(portfolio)

            result = execute_daily_job(settings, portfolio)
            signals = result["signals"]
            proposals = result["proposals"]
            data_keys = result["data_keys"]
            missing_symbols = result["missing_symbols"]
            selftest = result["selftest"]

            if not selftest["ok"]:
                st.error("Andmeallika enesetest kukkus läbi. Scan peatati.")
            else:
                if not signals.empty and signals["reason"].astype(str).str.contains("benchmark unavailable", case=False, na=False).any():
                    st.warning("Benchmark ei laadinud ära. Scan jooksis edasi NEUTRAL režiimis.")

                st.success(
                    f"Skänn tehtud. Turu režiim: {result['regime']}. "
                    f"Rangus: {get_strictness_label(signal_strictness)}. "
                    f"Teavitused: {result['notification_message']}"
                )
        except Exception as exc:
            st.error(f"Skänn ebaõnnestus: {exc}")

    render_selftest(selftest)

    if HISTORY_FILE.exists() and signals.empty:
        try:
            signals = pd.read_csv(HISTORY_FILE)
            proposals = build_actionable_proposals(portfolio, settings, signals)
        except Exception:
            pass

    st.subheader("Portfelli seis")
    st.dataframe(portfolio_snapshot_df(portfolio), use_container_width=True, hide_index=True)

    if not signals.empty:
        render_signal_summary(signals)
        render_skip_reasons(signals)
        render_top_signals(signals)
        render_loaded_missing(data_keys, missing_symbols)
    elif HISTORY_FILE.exists():
        try:
            hist = pd.read_csv(HISTORY_FILE)
            render_signal_summary(hist)
            render_skip_reasons(hist)
            render_top_signals(hist)
        except Exception:
            pass

    if run_scan_now or not proposals.empty:
        render_proposals(proposals, portfolio)


if __name__ == "__main__":
    run_streamlit_app()
