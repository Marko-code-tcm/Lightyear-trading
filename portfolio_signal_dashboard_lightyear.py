{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
"""\
Portfolio Signal Dashboard (Lightyear-aware)\
\
Features\
--------\
- Streamlit dashboard for a model portfolio\
- Investable capital input\
- Daily BUY / HOLD / SELL proposals\
- One-click acceptance of proposals into portfolio state\
- Persistent local state in JSON/CSV files\
- Automatic scheduled daily scan\
- Telegram or SMTP email notifications\
- Lightyear-only tradable universe filter via whitelist CSV\
- Simple lead-lag bonus signal between securities\
\
Install\
-------\
pip install streamlit pandas numpy yfinance apscheduler pytz\
\
Run dashboard\
-------------\
streamlit run portfolio_signal_dashboard_lightyear.py\
\
Run daily job once\
------------------\
python portfolio_signal_dashboard_lightyear.py --run-job\
\
Run scheduler in background/terminal\
------------------------------------\
python portfolio_signal_dashboard_lightyear.py --scheduler --hour 18 --minute 10 --timezone Europe/Tallinn\
\
Lightyear universe file\
-----------------------\
Create portfolio_state/lightyear_universe.csv with a column named symbol:\
\
symbol\
AAPL\
MSFT\
NVDA\
VWCE\
EUNL\
CNDX\
...\
\
The app will only propose symbols present in that file when strict Lightyear filtering is enabled.\
\
Notes\
-----\
- This is a model portfolio, not broker execution.\
- yfinance is suitable for prototyping and research, not institutional production.\
"""\
\
from __future__ import annotations\
\
import argparse\
import json\
import math\
import smtplib\
from dataclasses import dataclass\
from datetime import date, datetime\
from email.mime.text import MIMEText\
from pathlib import Path\
from typing import Dict, List, Optional, Tuple\
from urllib import parse, request\
\
import numpy as np\
import pandas as pd\
import pytz\
import streamlit as st\
import yfinance as yf\
from apscheduler.schedulers.blocking import BlockingScheduler\
\
\
APP_DIR = Path(".")\
STATE_DIR = APP_DIR / "portfolio_state"\
STATE_DIR.mkdir(parents=True, exist_ok=True)\
\
PORTFOLIO_FILE = STATE_DIR / "portfolio.json"\
SETTINGS_FILE = STATE_DIR / "settings.json"\
HISTORY_FILE = STATE_DIR / "signal_history.csv"\
TRANSACTIONS_FILE = STATE_DIR / "transactions.csv"\
LIGHTYEAR_UNIVERSE_FILE = STATE_DIR / "lightyear_universe.csv"\
LEADLAG_FILE = STATE_DIR / "leadlag_relationships.csv"\
\
DEFAULT_BENCHMARK = "SPY"\
DEFAULT_PERIOD = "18mo"\
DEFAULT_INTERVAL = "1d"\
DEFAULT_TIMEZONE = "Europe/Tallinn"\
DEFAULT_POSITION_MODE = "Equal weight"\
DEFAULT_MAX_NEW_POSITIONS_PER_DAY = 3\
DEFAULT_MIN_CASH_BUFFER_PCT = 0.05\
\
DEFAULT_LIGHTYEAR_SAFE_UNIVERSE = [\
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",\
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",\
    "JPM", "BAC", "GS", "XOM", "CVX", "LLY", "UNH", "COST", "CRM", "ORCL", "ADBE",\
    "VWCE", "VUSA", "CSPX", "EUNL", "SXR8", "VUAA", "CNDX", "IUIT",\
]\
\
\
@dataclass\
class StrategyConfig:\
    benchmark: str = DEFAULT_BENCHMARK\
    min_history_rows: int = 120\
    atr_stop_multiple: float = 2.5\
\
    regime_fast_ma: int = 50\
    regime_slow_ma: int = 200\
\
    sma_fast: int = 20\
    sma_mid: int = 50\
    sma_slow: int = 200\
    breakout_lookback: int = 20\
    momentum_lookback: int = 63\
    volume_lookback: int = 20\
    atr_lookback: int = 14\
    rs_lookback: int = 63\
\
    min_price: float = 5.0\
    min_avg_dollar_volume: float = 10_000_000.0\
    min_score_buy: float = 4.0\
    max_score_sell: float = -3.0\
\
    leadlag_enabled: bool = True\
    leadlag_train_window: int = 252\
    leadlag_max_lag: int = 3\
    leadlag_min_abs_corr: float = 0.25\
    leadlag_min_obs: int = 80\
    leadlag_min_t_stat: float = 2.0\
    leadlag_bonus_scale: float = 1.25\
\
\
def default_settings() -> Dict:\
    return \{\
        "investable_amount": 10000.0,\
        "user_universe": DEFAULT_LIGHTYEAR_SAFE_UNIVERSE,\
        "benchmark": DEFAULT_BENCHMARK,\
        "position_mode": DEFAULT_POSITION_MODE,\
        "max_new_positions_per_day": DEFAULT_MAX_NEW_POSITIONS_PER_DAY,\
        "min_cash_buffer_pct": DEFAULT_MIN_CASH_BUFFER_PCT,\
        "last_scan_date": None,\
        "auto_scan_enabled": True,\
        "scheduled_hour": 18,\
        "scheduled_minute": 10,\
        "scheduled_timezone": DEFAULT_TIMEZONE,\
        "notifications_enabled": False,\
        "notification_channel": "Telegram",\
        "telegram_bot_token": "",\
        "telegram_chat_id": "",\
        "smtp_host": "smtp.gmail.com",\
        "smtp_port": 587,\
        "smtp_username": "",\
        "smtp_password": "",\
        "smtp_to": "",\
        "leadlag_enabled": True,\
        "strict_lightyear_only": True,\
    \}\
\
\
def default_portfolio() -> Dict:\
    return \{\
        "cash": 10000.0,\
        "initial_capital": 10000.0,\
        "positions": \{\},\
        "closed_positions": [],\
        "last_updated": None,\
    \}\
\
\
def read_json(path: Path, fallback: Dict) -> Dict:\
    if not path.exists():\
        return fallback\
    try:\
        return json.loads(path.read_text(encoding="utf-8"))\
    except Exception:\
        return fallback\
\
\
def write_json(path: Path, payload: Dict) -> None:\
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\
\
\
def load_settings() -> Dict:\
    return default_settings() | read_json(SETTINGS_FILE, default_settings())\
\
\
def save_settings(settings: Dict) -> None:\
    write_json(SETTINGS_FILE, settings)\
\
\
def load_portfolio() -> Dict:\
    portfolio = default_portfolio() | read_json(PORTFOLIO_FILE, default_portfolio())\
    if "positions" not in portfolio or not isinstance(portfolio["positions"], dict):\
        portfolio["positions"] = \{\}\
    if "closed_positions" not in portfolio or not isinstance(portfolio["closed_positions"], list):\
        portfolio["closed_positions"] = []\
    return portfolio\
\
\
def save_portfolio(portfolio: Dict) -> None:\
    portfolio["last_updated"] = datetime.now().isoformat()\
    write_json(PORTFOLIO_FILE, portfolio)\
\
\
def append_csv_row(path: Path, row: Dict) -> None:\
    df_new = pd.DataFrame([row])\
    if path.exists():\
        df_old = pd.read_csv(path)\
        df = pd.concat([df_old, df_new], ignore_index=True)\
    else:\
        df = df_new\
    df.to_csv(path, index=False)\
\
\
def load_lightyear_universe() -> List[str]:\
    if LIGHTYEAR_UNIVERSE_FILE.exists():\
        try:\
            df = pd.read_csv(LIGHTYEAR_UNIVERSE_FILE)\
            possible_cols = [c for c in df.columns if c.lower() in \{"symbol", "ticker", "instrument"\}]\
            if possible_cols:\
                vals = [str(x).strip().upper() for x in df[possible_cols[0]].dropna().tolist() if str(x).strip()]\
                vals = list(dict.fromkeys(vals))\
                if vals:\
                    return vals\
        except Exception:\
            pass\
    return DEFAULT_LIGHTYEAR_SAFE_UNIVERSE.copy()\
\
\
def effective_scan_universe(settings: Dict) -> Tuple[List[str], List[str]]:\
    requested = [str(x).strip().upper() for x in settings.get("user_universe", []) if str(x).strip()]\
    lightyear = load_lightyear_universe()\
    lightyear_set = set(lightyear)\
    strict = bool(settings.get("strict_lightyear_only", True))\
\
    if strict:\
        filtered = [x for x in requested if x in lightyear_set]\
    else:\
        filtered = requested\
\
    if not filtered:\
        filtered = lightyear.copy()\
    return filtered, lightyear\
\
\
def download_symbol_history(symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:\
    try:\
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)\
    except Exception:\
        return pd.DataFrame()\
\
    if df is None or df.empty:\
        return pd.DataFrame()\
\
    if isinstance(df.columns, pd.MultiIndex):\
        df.columns = ["_".join([str(x) for x in tup if str(x)]).strip("_") for tup in df.columns]\
\
    cols = \{c.lower(): c for c in df.columns\}\
    required = ["open", "high", "low", "close", "volume"]\
    if not all(k in cols for k in required):\
        return pd.DataFrame()\
\
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()\
    out.columns = ["Open", "High", "Low", "Close", "Volume"]\
    out.index = pd.to_datetime(out.index)\
    return out.sort_index().dropna()\
\
\
def download_universe_data(symbols: List[str], period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> Dict[str, pd.DataFrame]:\
    out: Dict[str, pd.DataFrame] = \{\}\
    for symbol in sorted(set(symbols)):\
        df = download_symbol_history(symbol, period=period, interval=interval)\
        if not df.empty:\
            out[symbol] = df\
    return out\
\
\
def compute_atr(df: pd.DataFrame, lookback: int) -> pd.Series:\
    prev_close = df["Close"].shift(1)\
    tr = pd.concat([\
        df["High"] - df["Low"],\
        (df["High"] - prev_close).abs(),\
        (df["Low"] - prev_close).abs(),\
    ], axis=1).max(axis=1)\
    return tr.rolling(window=lookback, min_periods=lookback).mean()\
\
\
def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:\
    x = df.copy()\
    x["Ret1"] = x["Close"].pct_change()\
    x["SMA20"] = x["Close"].rolling(cfg.sma_fast).mean()\
    x["SMA50"] = x["Close"].rolling(cfg.sma_mid).mean()\
    x["SMA200"] = x["Close"].rolling(cfg.sma_slow).mean()\
    x["AvgVol20"] = x["Volume"].rolling(cfg.volume_lookback).mean()\
    x["DollarVol20"] = x["Close"] * x["AvgVol20"]\
    x["ATR14"] = compute_atr(x, cfg.atr_lookback)\
    x["High20"] = x["High"].rolling(cfg.breakout_lookback).max().shift(1)\
    x["Low20"] = x["Low"].rolling(cfg.breakout_lookback).min().shift(1)\
    x["Mom63"] = x["Close"] / x["Close"].shift(cfg.momentum_lookback) - 1.0\
    return x\
\
\
def compute_market_regime(benchmark_df: pd.DataFrame, cfg: StrategyConfig) -> str:\
    x = benchmark_df.copy()\
    x["SMA50"] = x["Close"].rolling(cfg.regime_fast_ma).mean()\
    x["SMA200"] = x["Close"].rolling(cfg.regime_slow_ma).mean()\
    latest = x.iloc[-1]\
    if pd.isna(latest["SMA50"]) or pd.isna(latest["SMA200"]):\
        return "UNKNOWN"\
    if latest["Close"] > latest["SMA200"] and latest["SMA50"] > latest["SMA200"]:\
        return "BULL"\
    if latest["Close"] < latest["SMA200"] and latest["SMA50"] < latest["SMA200"]:\
        return "BEAR"\
    return "NEUTRAL"\
\
\
def relative_strength_score(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame, lookback: int) -> float:\
    joined = pd.concat([\
        symbol_df[["Close"]].rename(columns=\{"Close": "sym_close"\}),\
        benchmark_df[["Close"]].rename(columns=\{"Close": "bench_close"\}),\
    ], axis=1, join="inner").dropna()\
    if len(joined) < lookback + 5:\
        return np.nan\
    rs = joined["sym_close"] / joined["bench_close"]\
    return float(rs.iloc[-1] / rs.iloc[-lookback] - 1.0)\
\
\
def build_returns_panel(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:\
    cols = \{\}\
    for symbol, df in data.items():\
        if "Close" in df.columns and len(df) > 10:\
            cols[symbol] = df["Close"].pct_change()\
    if not cols:\
        return pd.DataFrame()\
    return pd.DataFrame(cols).dropna(how="all")\
\
\
def discover_lead_lag_relationships(returns_panel: pd.DataFrame, symbols: List[str], cfg: StrategyConfig) -> pd.DataFrame:\
    if returns_panel.empty:\
        return pd.DataFrame(columns=["leader", "follower", "lag", "corr", "direction", "t_stat", "sign_consistency"])\
\
    syms = [s for s in symbols if s in returns_panel.columns]\
    panel = returns_panel[syms].tail(cfg.leadlag_train_window).dropna(how="all")\
    relationships: List[Dict] = []\
\
    for follower in syms:\
        best = None\
        follower_series = panel[follower]\
        for leader in syms:\
            if leader == follower:\
                continue\
            leader_series = panel[leader]\
            for lag in range(1, cfg.leadlag_max_lag + 1):\
                joined = pd.concat([\
                    leader_series.shift(lag).rename("leader"),\
                    follower_series.rename("follower"),\
                ], axis=1).dropna()\
                n = len(joined)\
                if n < cfg.leadlag_min_obs:\
                    continue\
\
                corr = joined["leader"].corr(joined["follower"])\
                if pd.isna(corr) or abs(corr) < cfg.leadlag_min_abs_corr or abs(corr) >= 0.999:\
                    continue\
\
                t_stat = abs(corr) * np.sqrt((n - 2) / max(1e-9, 1 - corr ** 2))\
                if t_stat < cfg.leadlag_min_t_stat:\
                    continue\
\
                mid = n // 2\
                first = joined.iloc[:mid]\
                second = joined.iloc[mid:]\
                if len(first) < max(20, cfg.leadlag_min_obs // 3) or len(second) < max(20, cfg.leadlag_min_obs // 3):\
                    continue\
\
                corr_first = first["leader"].corr(first["follower"])\
                corr_second = second["leader"].corr(second["follower"])\
                if pd.isna(corr_first) or pd.isna(corr_second):\
                    continue\
                sign_consistency = int(np.sign(corr_first) == np.sign(corr_second) == np.sign(corr))\
                if not sign_consistency:\
                    continue\
\
                candidate = \{\
                    "leader": leader,\
                    "follower": follower,\
                    "lag": lag,\
                    "corr": float(corr),\
                    "direction": "same" if corr > 0 else "opposite",\
                    "t_stat": round(float(t_stat), 3),\
                    "sign_consistency": sign_consistency,\
                \}\
                if best is None or abs(candidate["corr"]) > abs(best["corr"]):\
                    best = candidate\
        if best is not None:\
            relationships.append(best)\
\
    return pd.DataFrame(relationships)\
\
\
def compute_leadlag_signal(symbol: str, relationships: pd.DataFrame, returns_panel: pd.DataFrame, cfg: StrategyConfig) -> Tuple[float, str]:\
    if relationships.empty or returns_panel.empty:\
        return 0.0, ""\
    rows = relationships[relationships["follower"] == symbol]\
    if rows.empty:\
        return 0.0, ""\
\
    best_bonus = 0.0\
    best_reason = ""\
    for _, rel in rows.iterrows():\
        leader = str(rel["leader"])\
        lag = int(rel["lag"])\
        corr = float(rel["corr"])\
        if leader not in returns_panel.columns:\
            continue\
\
        leader_series = returns_panel[leader].dropna()\
        if len(leader_series) < lag + 20:\
            continue\
\
        leader_ret = float(leader_series.iloc[-lag])\
        recent_window = leader_series.tail(60)\
        vol = float(recent_window.std()) if len(recent_window) >= 20 else float(leader_series.std())\
        if not np.isfinite(vol) or vol <= 0:\
            continue\
\
        z = leader_ret / vol\
        if abs(z) < 1.0:\
            continue\
\
        raw_bonus = np.sign(leader_ret) * np.sign(corr) * min(abs(corr), 0.4) * cfg.leadlag_bonus_scale\
        bonus = float(np.sign(raw_bonus) * min(abs(raw_bonus), 0.5))\
        if abs(bonus) > abs(best_bonus):\
            best_bonus = bonus\
            arrow = "\uc0\u8593 " if leader_ret > 0 else "\u8595 "\
            direction = "same direction" if corr > 0 else "opposite direction"\
            best_reason = f"strict lead-lag: \{leader\} \{arrow\} \{lag\}d ago on an outsized move (z\uc0\u8776 \{z:.2f\}); follower tends to move \{direction\}"\
\
    return round(best_bonus, 3), best_reason\
\
\
def score_symbol(symbol: str, df: pd.DataFrame, benchmark_df: pd.DataFrame, regime: str, cfg: StrategyConfig,\
                 leadlag_bonus: float = 0.0, leadlag_reason: str = "") -> Dict:\
    x = add_indicators(df, cfg)\
    if len(x) < cfg.min_history_rows:\
        return \{"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "not enough history"\}\
\
    latest = x.iloc[-1]\
    prev = x.iloc[-2] if len(x) >= 2 else latest\
    price = float(latest["Close"])\
    avg_dollar_vol = float(latest["DollarVol20"]) if pd.notna(latest["DollarVol20"]) else np.nan\
\
    if price < cfg.min_price:\
        return \{"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "price too low"\}\
    if np.isnan(avg_dollar_vol) or avg_dollar_vol < cfg.min_avg_dollar_volume:\
        return \{"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "insufficient liquidity"\}\
\
    rs = relative_strength_score(x, benchmark_df, cfg.rs_lookback)\
    atr = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else np.nan\
    atr_pct = float(atr / price) if price > 0 and pd.notna(atr) else np.nan\
    mom63 = float(latest["Mom63"]) if pd.notna(latest["Mom63"]) else np.nan\
\
    score = 0.0\
    reasons: List[str] = []\
\
    if pd.notna(latest["SMA20"]) and price > latest["SMA20"]:\
        score += 1.0\
        reasons.append("above SMA20")\
    else:\
        score -= 1.0\
        reasons.append("below SMA20")\
\
    if pd.notna(latest["SMA50"]) and price > latest["SMA50"]:\
        score += 1.5\
        reasons.append("above SMA50")\
    else:\
        score -= 1.5\
        reasons.append("below SMA50")\
\
    if pd.notna(latest["SMA200"]) and price > latest["SMA200"]:\
        score += 2.0\
        reasons.append("above SMA200")\
    else:\
        score -= 2.0\
        reasons.append("below SMA200")\
\
    if pd.notna(mom63):\
        if mom63 > 0.15:\
            score += 2.0\
            reasons.append("strong 3m momentum")\
        elif mom63 > 0.05:\
            score += 1.0\
            reasons.append("positive 3m momentum")\
        elif mom63 < -0.10:\
            score -= 2.0\
            reasons.append("weak 3m momentum")\
        elif mom63 < 0:\
            score -= 1.0\
            reasons.append("negative 3m momentum")\
\
    if pd.notna(rs):\
        if rs > 0.10:\
            score += 1.5\
            reasons.append("outperforming benchmark")\
        elif rs < -0.05:\
            score -= 1.5\
            reasons.append("underperforming benchmark")\
\
    if pd.notna(latest["High20"]) and price > latest["High20"]:\
        score += 1.5\
        reasons.append("20d breakout")\
    if pd.notna(latest["Low20"]) and price < latest["Low20"]:\
        score -= 1.5\
        reasons.append("20d breakdown")\
\
    vol_ratio = np.nan\
    if pd.notna(latest["AvgVol20"]) and latest["AvgVol20"] > 0:\
        vol_ratio = float(latest["Volume"] / latest["AvgVol20"])\
        if vol_ratio > 1.3 and price > float(prev["Close"]):\
            score += 1.0\
            reasons.append("volume confirms up move")\
        elif vol_ratio > 1.3 and price < float(prev["Close"]):\
            score -= 1.0\
            reasons.append("volume confirms down move")\
\
    if pd.notna(atr_pct):\
        if atr_pct > 0.08:\
            score -= 1.0\
            reasons.append("high ATR risk")\
        elif atr_pct < 0.03:\
            score += 0.5\
            reasons.append("contained ATR")\
\
    if regime == "BULL":\
        if score > 0:\
            score += 0.5\
            reasons.append("bull market tailwind")\
    elif regime == "BEAR":\
        if score > 0:\
            score -= 2.0\
            reasons.append("bear market penalty")\
        else:\
            score -= 0.5\
            reasons.append("bear market pressure")\
\
    score += leadlag_bonus\
    if abs(leadlag_bonus) > 0:\
        reasons.append(leadlag_reason)\
\
    action = "HOLD"\
    if score >= cfg.min_score_buy and regime != "BEAR":\
        action = "BUY"\
    elif score <= cfg.max_score_sell:\
        action = "SELL"\
\
    stop_ref = np.nan\
    if pd.notna(atr):\
        stop_ref = price - cfg.atr_stop_multiple * atr if action == "BUY" else price + cfg.atr_stop_multiple * atr\
\
        aligned = 0\
    if action == "BUY" and leadlag_bonus > 0:\
        aligned = 1\
    elif action == "SELL" and leadlag_bonus < 0:\
        aligned = 1\
\
    confidence = "Low"\
    conviction = "Normal"\
    abs_score = abs(float(score))\
    abs_ll = abs(float(leadlag_bonus))\
\
    if abs_score >= 6.5 and aligned and abs_ll >= 0.2:\
        confidence = "High"\
        conviction = "High conviction"\
    elif abs_score >= 5.0:\
        confidence = "Medium"\
    else:\
        confidence = "Low"\
\
    return \{\
        "symbol": symbol,\
        "action": action,\
        "score": round(float(score), 3),\
        "leadlag_bonus": round(float(leadlag_bonus), 3),\
        "close": round(price, 4),\
        "mom63": round(mom63, 4) if pd.notna(mom63) else np.nan,\
        "rs_vs_benchmark": round(rs, 4) if pd.notna(rs) else np.nan,\
        "atr_pct": round(atr_pct, 4) if pd.notna(atr_pct) else np.nan,\
        "volume_ratio": round(vol_ratio, 4) if pd.notna(vol_ratio) else np.nan,\
        "avg_dollar_vol_20": round(avg_dollar_vol, 2) if pd.notna(avg_dollar_vol) else np.nan,\
        "stop_reference": round(float(stop_ref), 4) if pd.notna(stop_ref) else np.nan,\
        "confidence": confidence,\
        "conviction": conviction,\
        "reason": "; ".join(reasons[:10]),\
    \}\
\
\
def run_daily_scan(universe: List[str], benchmark: str, leadlag_enabled: bool = True) -> Tuple[str, pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:\
    cfg = StrategyConfig(benchmark=benchmark, leadlag_enabled=leadlag_enabled)\
    symbols = sorted(set(universe + [benchmark]))\
    data = download_universe_data(symbols)\
    if benchmark not in data:\
        raise RuntimeError(f"Benchmark data missing for \{benchmark\}")\
\
    benchmark_df = data[benchmark]\
    regime = compute_market_regime(benchmark_df, cfg)\
    returns_panel = build_returns_panel(data)\
    relationships = discover_lead_lag_relationships(returns_panel, universe, cfg) if leadlag_enabled else pd.DataFrame()\
\
    rows: List[Dict] = []\
    for symbol in universe:\
        df = data.get(symbol)\
        if df is None or df.empty:\
            rows.append(\{"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": "missing data"\})\
            continue\
        leadlag_bonus, leadlag_reason = (0.0, "")\
        if leadlag_enabled:\
            leadlag_bonus, leadlag_reason = compute_leadlag_signal(symbol, relationships, returns_panel, cfg)\
        try:\
            rows.append(score_symbol(symbol, df, benchmark_df, regime, cfg, leadlag_bonus, leadlag_reason))\
        except Exception as exc:\
            rows.append(\{"symbol": symbol, "action": "SKIP", "score": np.nan, "reason": f"error: \{exc\}"\})\
\
    signals = pd.DataFrame(rows)\
    if not signals.empty:\
        signals["scan_date"] = pd.Timestamp(date.today())\
        signals = signals.sort_values(["action", "score"], ascending=[True, False])\
    return regime, signals, data, relationships\
\
\
def normalize_positions(portfolio: Dict) -> None:\
    for symbol, pos in portfolio["positions"].items():\
        pos.setdefault("symbol", symbol)\
        pos.setdefault("quantity", 0)\
        pos.setdefault("avg_price", 0.0)\
        pos.setdefault("market_value", 0.0)\
        pos.setdefault("last_price", 0.0)\
        pos.setdefault("status", "HOLD")\
        pos.setdefault("stop_reference", None)\
        pos.setdefault("opened_at", datetime.now().isoformat())\
        pos.setdefault("last_signal_score", None)\
        pos.setdefault("last_signal_reason", "")\
\
\
def reset_portfolio_for_new_capital(settings: Dict, portfolio: Dict) -> None:\
    investable = float(settings["investable_amount"])\
    if not portfolio.get("positions") and not portfolio.get("closed_positions"):\
        portfolio["cash"] = investable\
        portfolio["initial_capital"] = investable\
\
\
def estimate_portfolio_value(portfolio: Dict) -> float:\
    total = float(portfolio.get("cash", 0.0))\
    for pos in portfolio.get("positions", \{\}).values():\
        total += float(pos.get("market_value", 0.0))\
    return total\
\
\
def update_position_marks(portfolio: Dict, signals: pd.DataFrame) -> None:\
    normalize_positions(portfolio)\
    signal_map = \{row["symbol"]: row for _, row in signals.iterrows()\}\
    for symbol, pos in portfolio["positions"].items():\
        row = signal_map.get(symbol)\
        if row is None:\
            continue\
        last_price = float(row.get("close", pos.get("last_price", 0.0)))\
        qty = float(pos.get("quantity", 0.0))\
        pos["last_price"] = last_price\
        pos["market_value"] = round(last_price * qty, 2)\
        pos["status"] = "SELL" if row.get("action") == "SELL" else "HOLD"\
        pos["last_signal_score"] = None if pd.isna(row.get("score")) else float(row.get("score"))\
        pos["last_signal_reason"] = str(row.get("reason", ""))\
        pos["stop_reference"] = None if pd.isna(row.get("stop_reference")) else float(row.get("stop_reference"))\
\
\
def current_open_symbols(portfolio: Dict) -> set:\
    return set(portfolio.get("positions", \{\}).keys())\
\
\
def available_cash_for_new_positions(portfolio: Dict, settings: Dict) -> float:\
    cash = float(portfolio.get("cash", 0.0))\
    min_buffer = estimate_portfolio_value(portfolio) * float(settings.get("min_cash_buffer_pct", DEFAULT_MIN_CASH_BUFFER_PCT))\
    return max(0.0, cash - min_buffer)\
\
\
def determine_buy_budget(portfolio: Dict, settings: Dict, num_candidates: int) -> float:\
    if num_candidates <= 0:\
        return 0.0\
    free_cash = available_cash_for_new_positions(portfolio, settings)\
    if free_cash <= 0:\
        return 0.0\
\
    mode = settings.get("position_mode", DEFAULT_POSITION_MODE)\
    slots = max(1, min(num_candidates, int(settings.get("max_new_positions_per_day", DEFAULT_MAX_NEW_POSITIONS_PER_DAY))))\
\
    if mode == "Equal weight":\
        current_positions = len(portfolio.get("positions", \{\}))\
        target_positions = max(current_positions + slots, 1)\
        target_size = estimate_portfolio_value(portfolio) / target_positions\
        return max(0.0, min(free_cash / slots, target_size))\
\
    return free_cash / slots\
\
\
def build_actionable_proposals(portfolio: Dict, settings: Dict, signals: pd.DataFrame) -> pd.DataFrame:\
    if signals.empty:\
        return pd.DataFrame()\
\
    open_symbols = current_open_symbols(portfolio)\
    rows = []\
\
    buys = signals[(signals["action"] == "BUY") & (~signals["symbol"].isin(open_symbols))].copy()\
    buys = buys.sort_values("score", ascending=False)\
    budget_per_buy = determine_buy_budget(portfolio, settings, len(buys))\
\
    for _, row in buys.iterrows():\
        price = float(row["close"])\
        qty = int(math.floor(budget_per_buy / price)) if price > 0 else 0\
        est_cost = round(qty * price, 2)\
        if qty <= 0:\
            continue\
        rows.append(\{**row.to_dict(), "proposal_type": "BUY", "quantity": qty, "estimated_cash_impact": -est_cost, "note": "Open new position"\})\
\
    for _, row in signals.iterrows():\
        symbol = str(row["symbol"])\
        if symbol not in open_symbols:\
            continue\
        pos = portfolio["positions"][symbol]\
        qty = int(pos.get("quantity", 0))\
        price = float(row["close"])\
        proposal = "SELL" if row["action"] == "SELL" else "HOLD"\
        rows.append(\{\
            **row.to_dict(),\
            "proposal_type": proposal,\
            "quantity": qty,\
            "estimated_cash_impact": round(qty * price, 2) if proposal == "SELL" else 0.0,\
            "note": "Close position" if proposal == "SELL" else "Keep holding",\
        \})\
\
    proposals = pd.DataFrame(rows)\
    if proposals.empty:\
        return proposals\
    order_map = \{"BUY": 0, "SELL": 1, "HOLD": 2\}\
    proposals["order_key"] = proposals["proposal_type"].map(order_map)\
    return proposals.sort_values(["order_key", "score"], ascending=[True, False]).drop(columns=["order_key"])\
\
\
def accept_buy(portfolio: Dict, symbol: str, quantity: int, price: float, reason: str, score: Optional[float], stop_reference: Optional[float]) -> str:\
    cost = round(quantity * price, 2)\
    cash = float(portfolio.get("cash", 0.0))\
    if quantity <= 0:\
        return "Kogus peab olema suurem kui 0."\
    if cost > cash:\
        return f"Vaba raha ei piisa. Vajalik \{cost:.2f\}, olemas \{cash:.2f\}."\
    if symbol in portfolio.get("positions", \{\}):\
        return f"\{symbol\} on juba portfellis."\
\
    portfolio["cash"] = round(cash - cost, 2)\
    portfolio["positions"][symbol] = \{\
        "symbol": symbol,\
        "quantity": int(quantity),\
        "avg_price": round(price, 4),\
        "last_price": round(price, 4),\
        "market_value": round(cost, 2),\
        "status": "HOLD",\
        "stop_reference": None if stop_reference is None or pd.isna(stop_reference) else round(float(stop_reference), 4),\
        "opened_at": datetime.now().isoformat(),\
        "last_signal_score": None if score is None or pd.isna(score) else float(score),\
        "last_signal_reason": reason,\
    \}\
    append_csv_row(TRANSACTIONS_FILE, \{"timestamp": datetime.now().isoformat(), "type": "BUY", "symbol": symbol, "quantity": quantity, "price": round(price, 4), "gross_amount": cost, "reason": reason\})\
    save_portfolio(portfolio)\
    return f"Ost aktsepteeritud: \{symbol\}, \{quantity\} tk hinnaga \{price:.2f\}."\
\
\
def accept_sell(portfolio: Dict, symbol: str, price: float, reason: str, score: Optional[float]) -> str:\
    pos = portfolio.get("positions", \{\}).get(symbol)\
    if not pos:\
        return f"\{symbol\} ei ole portfellis."\
    qty = int(pos.get("quantity", 0))\
    if qty <= 0:\
        return f"\{symbol\} kogus on vigane."\
\
    proceeds = round(qty * price, 2)\
    avg_price = float(pos.get("avg_price", 0.0))\
    pnl = round((price - avg_price) * qty, 2)\
    portfolio["cash"] = round(float(portfolio.get("cash", 0.0)) + proceeds, 2)\
    portfolio["closed_positions"].append(\{**pos, "closed_at": datetime.now().isoformat(), "exit_price": round(price, 4), "realized_pnl": pnl, "close_reason": reason, "last_signal_score": None if score is None or pd.isna(score) else float(score)\})\
    del portfolio["positions"][symbol]\
    append_csv_row(TRANSACTIONS_FILE, \{"timestamp": datetime.now().isoformat(), "type": "SELL", "symbol": symbol, "quantity": qty, "price": round(price, 4), "gross_amount": proceeds, "reason": reason, "realized_pnl": pnl\})\
    save_portfolio(portfolio)\
    return f"M\'fc\'fck aktsepteeritud: \{symbol\}, \{qty\} tk hinnaga \{price:.2f\}, P/L \{pnl:.2f\}."\
\
\
def record_hold_review(symbol: str, reason: str, score: Optional[float]) -> None:\
    append_csv_row(TRANSACTIONS_FILE, \{"timestamp": datetime.now().isoformat(), "type": "HOLD_REVIEW", "symbol": symbol, "quantity": None, "price": None, "gross_amount": None, "reason": reason, "score": None if score is None or pd.isna(score) else float(score)\})\
\
\
def build_notification_message(regime: str, proposals: pd.DataFrame, settings: Dict) -> str:\
    lines = [f"Daily portfolio scan \'97 \{date.today().isoformat()\}", f"Benchmark: \{settings.get('benchmark')\} | Regime: \{regime\}", ""]\
    for label in ["BUY", "SELL", "HOLD"]:\
        subset = proposals[proposals["proposal_type"] == label]\
        lines.append(label)\
        if subset.empty:\
            lines.append("- none")\
        else:\
            limit = 5 if label != "HOLD" else 3\
            for _, row in subset.head(limit).iterrows():\
                qty_text = f" | qty \{int(row['quantity'])\}" if pd.notna(row.get("quantity")) else ""\
                ll = row.get("leadlag_bonus")\
                ll_text = f" | lead-lag \{float(ll):+.2f\}" if pd.notna(ll) else ""\
                conf_text = f" | \{row.get('confidence', 'NA')\}"\
                conv = row.get('conviction', 'Normal')\
                conv_text = f" | \{conv\}" if conv and conv != 'Normal' else ""\
                lines.append(f"- \{row['symbol']\} @ \{float(row['close']):.2f\} | score \{float(row['score']):+.2f\}\{conf_text\}\{conv_text\}\{qty_text\}\{ll_text\}")\
        lines.append("")\
    return "\\n".join(lines).strip()\
\
\
def send_telegram_message(bot_token: str, chat_id: str, text: str) -> Tuple[bool, str]:\
    if not bot_token or not chat_id:\
        return False, "Telegram token or chat id is missing"\
    try:\
        url = f"https://api.telegram.org/bot\{bot_token\}/sendMessage"\
        payload = parse.urlencode(\{"chat_id": chat_id, "text": text\}).encode("utf-8")\
        req = request.Request(url, data=payload)\
        with request.urlopen(req, timeout=20) as resp:\
            return resp.status == 200, "Telegram sent" if resp.status == 200 else "Telegram failed"\
    except Exception as exc:\
        return False, f"Telegram error: \{exc\}"\
\
\
def send_email_message(host: str, port: int, username: str, password: str, to_addr: str, text: str) -> Tuple[bool, str]:\
    if not all([host, port, username, password, to_addr]):\
        return False, "SMTP settings are incomplete"\
    try:\
        msg = MIMEText(text, "plain", "utf-8")\
        msg["Subject"] = f"Daily portfolio scan \{date.today().isoformat()\}"\
        msg["From"] = username\
        msg["To"] = to_addr\
        with smtplib.SMTP(host, port, timeout=20) as server:\
            server.starttls()\
            server.login(username, password)\
            server.send_message(msg)\
        return True, "Email sent"\
    except Exception as exc:\
        return False, f"Email error: \{exc\}"\
\
\
def send_notifications(settings: Dict, regime: str, proposals: pd.DataFrame) -> Tuple[bool, str]:\
    if not settings.get("notifications_enabled", False):\
        return False, "Notifications are disabled"\
    text = build_notification_message(regime, proposals, settings)\
    channel = settings.get("notification_channel", "Telegram")\
    if channel == "Telegram":\
        return send_telegram_message(settings.get("telegram_bot_token", ""), settings.get("telegram_chat_id", ""), text)\
    return send_email_message(settings.get("smtp_host", "smtp.gmail.com"), int(settings.get("smtp_port", 587)), settings.get("smtp_username", ""), settings.get("smtp_password", ""), settings.get("smtp_to", ""), text)\
\
\
def execute_daily_job() -> Dict:\
    settings = load_settings()\
    portfolio = load_portfolio()\
    reset_portfolio_for_new_capital(settings, portfolio)\
    normalize_positions(portfolio)\
\
    universe, lightyear_universe = effective_scan_universe(settings)\
    benchmark = str(settings.get("benchmark", DEFAULT_BENCHMARK)).upper().strip()\
    if benchmark not in universe:\
        universe = universe + [benchmark]\
    if settings.get("strict_lightyear_only", True) and benchmark not in lightyear_universe:\
        lightyear_universe.append(benchmark)\
\
    regime, signals, _, relationships = run_daily_scan(universe=universe, benchmark=benchmark, leadlag_enabled=bool(settings.get("leadlag_enabled", True)))\
    settings["last_scan_date"] = str(date.today())\
    save_settings(settings)\
\
    update_position_marks(portfolio, signals)\
    save_portfolio(portfolio)\
    proposals = build_actionable_proposals(portfolio, settings, signals)\
\
    if not signals.empty:\
        export = signals.copy()\
        export["benchmark"] = benchmark\
        export["regime"] = regime\
        export.to_csv(HISTORY_FILE, index=False)\
    if not relationships.empty:\
        relationships.to_csv(LEADLAG_FILE, index=False)\
\
    notif_ok, notif_msg = send_notifications(settings, regime, proposals)\
    return \{"regime": regime, "signals": signals, "proposals": proposals, "relationships": relationships, "notification_ok": notif_ok, "notification_message": notif_msg, "scan_universe": universe\}\
\
\
def fmt_money(x: float) -> str:\
    return f"\{x:,.2f\}".replace(",", " ")\
\
\
def portfolio_snapshot_df(portfolio: Dict) -> pd.DataFrame:\
    rows = []\
    for symbol, pos in portfolio.get("positions", \{\}).items():\
        qty = float(pos.get("quantity", 0.0))\
        avg_price = float(pos.get("avg_price", 0.0))\
        last_price = float(pos.get("last_price", 0.0))\
        mv = float(pos.get("market_value", qty * last_price))\
        rows.append(\{\
            "symbol": symbol,\
            "quantity": int(qty),\
            "avg_price": round(avg_price, 4),\
            "last_price": round(last_price, 4),\
            "market_value": round(mv, 2),\
            "unrealized_pnl": round((last_price - avg_price) * qty, 2),\
            "status": pos.get("status", "HOLD"),\
            "stop_reference": pos.get("stop_reference"),\
            "reason": pos.get("last_signal_reason", ""),\
        \})\
    if not rows:\
        return pd.DataFrame(columns=["symbol", "quantity", "avg_price", "last_price", "market_value", "unrealized_pnl", "status", "stop_reference", "reason"])\
    return pd.DataFrame(rows).sort_values(["status", "market_value"], ascending=[True, False])\
\
\
def proposal_table_df(proposals: pd.DataFrame) -> pd.DataFrame:\
    if proposals.empty:\
        return proposals\
    cols = ["symbol", "proposal_type", "score", "confidence", "conviction", "leadlag_bonus", "close", "quantity", "estimated_cash_impact", "stop_reference", "reason", "note"]\
    return proposals[[c for c in cols if c in proposals.columns]].copy()\
\
\
def run_streamlit_app() -> None:\
    st.set_page_config(page_title="Portfolio Signal Dashboard", layout="wide")\
    st.title("Portfolio Signal Dashboard")\
    st.caption("Igap\'e4evane ostu-, hoidmis- ja m\'fc\'fcgiettepanekute t\'f6\'f6laud koos Lightyear-filtri, automaatk\'e4ivituse ja lead-lag lisasignaaliga.")\
\
    settings = load_settings()\
    portfolio = load_portfolio()\
    reset_portfolio_for_new_capital(settings, portfolio)\
    normalize_positions(portfolio)\
\
    filtered_universe, lightyear_universe = effective_scan_universe(settings)\
\
    with st.sidebar:\
        st.header("Seaded")\
        investable_amount = st.number_input("Investeeritav summa", min_value=0.0, value=float(settings.get("investable_amount", 10000.0)), step=100.0)\
        benchmark = st.text_input("Benchmark", value=str(settings.get("benchmark", DEFAULT_BENCHMARK))).upper().strip()\
        universe_text = st.text_area("J\'e4lgitavad tickerid (komaga eraldatud)", value=", ".join(settings.get("user_universe", DEFAULT_LIGHTYEAR_SAFE_UNIVERSE)), height=130)\
        position_mode = st.selectbox("Uute ostude jaotus", options=["Equal weight", "Use available cash evenly"], index=0 if settings.get("position_mode", DEFAULT_POSITION_MODE) == "Equal weight" else 1)\
        max_new_positions_per_day = st.number_input("Maksimaalne uute positsioonide arv p\'e4evas", min_value=1, max_value=20, value=int(settings.get("max_new_positions_per_day", DEFAULT_MAX_NEW_POSITIONS_PER_DAY)), step=1)\
        min_cash_buffer_pct = st.slider("Raha puhver osakaaluna portfellist", min_value=0.0, max_value=0.30, value=float(settings.get("min_cash_buffer_pct", DEFAULT_MIN_CASH_BUFFER_PCT)), step=0.01)\
        strict_lightyear_only = st.checkbox("Kasuta ainult Lightyear whitelist'i", value=bool(settings.get("strict_lightyear_only", True)))\
        leadlag_enabled = st.checkbox("Kasuta lead-lag lisasignaali", value=bool(settings.get("leadlag_enabled", True)))\
\
        st.subheader("Automaatk\'e4ivitus")\
        auto_scan_enabled = st.checkbox("Automaatne p\'e4evask\'e4nn lubatud", value=bool(settings.get("auto_scan_enabled", True)))\
        scheduled_hour = st.number_input("Tund", min_value=0, max_value=23, value=int(settings.get("scheduled_hour", 18)), step=1)\
        scheduled_minute = st.number_input("Minut", min_value=0, max_value=59, value=int(settings.get("scheduled_minute", 10)), step=1)\
        scheduled_timezone = st.text_input("Timezone", value=str(settings.get("scheduled_timezone", DEFAULT_TIMEZONE)))\
\
        st.subheader("Teavitused")\
        notifications_enabled = st.checkbox("Teavitused lubatud", value=bool(settings.get("notifications_enabled", False)))\
        notification_channel = st.selectbox("Kanal", options=["Telegram", "Email"], index=0 if settings.get("notification_channel", "Telegram") == "Telegram" else 1)\
\
        telegram_bot_token = st.text_input("Telegram bot token", value=str(settings.get("telegram_bot_token", "")), type="password")\
        telegram_chat_id = st.text_input("Telegram chat id", value=str(settings.get("telegram_chat_id", "")))\
        smtp_host = st.text_input("SMTP host", value=str(settings.get("smtp_host", "smtp.gmail.com")))\
        smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(settings.get("smtp_port", 587)), step=1)\
        smtp_username = st.text_input("SMTP username", value=str(settings.get("smtp_username", "")))\
        smtp_password = st.text_input("SMTP password", value=str(settings.get("smtp_password", "")), type="password")\
        smtp_to = st.text_input("SMTP to", value=str(settings.get("smtp_to", "")))\
\
        if st.button("Salvesta seaded", use_container_width=True):\
            parsed_universe = [x.strip().upper() for x in universe_text.split(",") if x.strip()]\
            settings.update(\{\
                "investable_amount": float(investable_amount),\
                "benchmark": benchmark,\
                "user_universe": parsed_universe,\
                "position_mode": position_mode,\
                "max_new_positions_per_day": int(max_new_positions_per_day),\
                "min_cash_buffer_pct": float(min_cash_buffer_pct),\
                "strict_lightyear_only": bool(strict_lightyear_only),\
                "leadlag_enabled": bool(leadlag_enabled),\
                "auto_scan_enabled": bool(auto_scan_enabled),\
                "scheduled_hour": int(scheduled_hour),\
                "scheduled_minute": int(scheduled_minute),\
                "scheduled_timezone": scheduled_timezone,\
                "notifications_enabled": bool(notifications_enabled),\
                "notification_channel": notification_channel,\
                "telegram_bot_token": telegram_bot_token,\
                "telegram_chat_id": telegram_chat_id,\
                "smtp_host": smtp_host,\
                "smtp_port": int(smtp_port),\
                "smtp_username": smtp_username,\
                "smtp_password": smtp_password,\
                "smtp_to": smtp_to,\
            \})\
            save_settings(settings)\
            if not portfolio.get("positions") and not portfolio.get("closed_positions"):\
                portfolio["cash"] = float(investable_amount)\
                portfolio["initial_capital"] = float(investable_amount)\
                save_portfolio(portfolio)\
            st.success("Seaded salvestatud.")\
\
        if st.button("Nulli portfell", use_container_width=True):\
            new_portfolio = default_portfolio()\
            new_portfolio["cash"] = float(investable_amount)\
            new_portfolio["initial_capital"] = float(investable_amount)\
            save_portfolio(new_portfolio)\
            st.warning("Portfell nulliti. Lae leht uuesti v\'f5i vajuta Run daily scan.")\
\
    c1, c2, c3, c4 = st.columns(4)\
    pv = estimate_portfolio_value(portfolio)\
    cash = float(portfolio.get("cash", 0.0))\
    c1.metric("Portfelli v\'e4\'e4rtus", fmt_money(pv))\
    c2.metric("Vaba raha", fmt_money(cash))\
    c3.metric("Investeeritud", fmt_money(pv - cash))\
    c4.metric("Avatud positsioone", str(len(portfolio.get("positions", \{\}))))\
\
    i1, i2, i3 = st.columns(3)\
    i1.info(f"Lightyear whitelist suurus: \{len(lightyear_universe)\}")\
    i2.info(f"Sk\'e4nnitav universum: \{len(filtered_universe)\}")\
    i3.info(f"Viimane scan: \{settings.get('last_scan_date') or 'puudub'\}")\
\
    st.caption("Kui soovid ranget Lightyear filtrit, pane t\'e4pne tickerite nimekiri faili portfolio_state/lightyear_universe.csv.")\
\
    run_scan_now = st.button("Run daily scan", type="primary", use_container_width=True)\
    signals = pd.DataFrame()\
    proposals = pd.DataFrame()\
    regime = None\
    relationships = pd.DataFrame()\
\
    if run_scan_now:\
        try:\
            result = execute_daily_job()\
            regime = result["regime"]\
            signals = result["signals"]\
            proposals = result["proposals"]\
            relationships = result["relationships"]\
            st.success(f"Sk\'e4nn tehtud. Turu re\'9eiim: \{regime\}. Teavitused: \{result['notification_message']\}")\
        except Exception as exc:\
            st.error(f"Sk\'e4nn eba\'f5nnestus: \{exc\}")\
\
    if HISTORY_FILE.exists() and signals.empty:\
        try:\
            signals = pd.read_csv(HISTORY_FILE)\
            regime_vals = signals["regime"].dropna().unique().tolist() if "regime" in signals.columns else []\
            regime = regime_vals[0] if regime_vals else None\
            update_position_marks(portfolio, signals)\
            save_portfolio(portfolio)\
            proposals = build_actionable_proposals(portfolio, settings, signals)\
        except Exception:\
            pass\
    if LEADLAG_FILE.exists() and relationships.empty:\
        try:\
            relationships = pd.read_csv(LEADLAG_FILE)\
        except Exception:\
            relationships = pd.DataFrame()\
\
    st.subheader("T\'e4nased ettepanekud")\
    if proposals.empty:\
        st.write("T\'e4na ettepanekuid veel ei ole. Vajuta **Run daily scan**.")\
    else:\
        st.dataframe(proposal_table_df(proposals), use_container_width=True, hide_index=True)\
        buy_tab, sell_tab, hold_tab = st.tabs(["BUY", "SELL", "HOLD"])\
\
        with buy_tab:\
            df = proposals[proposals["proposal_type"] == "BUY"].sort_values("score", ascending=False)\
            if df.empty:\
                st.write("T\'e4na uusi ostukandidaate ei ole.")\
            for _, row in df.iterrows():\
                sym = str(row["symbol"])\
                with st.container(border=True):\
                    a, b, c = st.columns([2, 2, 1])\
                    a.markdown(f"**\{sym\}**")\
                    a.write(f"Skoor: \{row['score']\}")\
                    a.write(f"Lead-lag boonus: \{row.get('leadlag_bonus', 0)\}")\
                    a.write(f"Confidence: \{row.get('confidence', 'NA')\} | \{row.get('conviction', 'Normal')\}")\
                    a.write(f"Hind: \{row['close']\}")\
                    b.write(f"Kogus: \{int(row['quantity'])\}")\
                    b.write(f"Eeldatav kulu: \{fmt_money(abs(float(row['estimated_cash_impact'])))\}")\
                    b.write(f"P\'f5hjus: \{row['reason']\}")\
                    if c.button(f"Aksepteeri BUY \{sym\}", key=f"buy_\{sym\}", use_container_width=True):\
                        st.success(accept_buy(portfolio, sym, int(row['quantity']), float(row['close']), str(row.get('reason', '')), None if pd.isna(row.get('score')) else float(row.get('score')), None if pd.isna(row.get('stop_reference')) else float(row.get('stop_reference'))))\
                        st.rerun()\
\
        with sell_tab:\
            df = proposals[proposals["proposal_type"] == "SELL"].sort_values("score", ascending=True)\
            if df.empty:\
                st.write("T\'e4na m\'fc\'fcgiettepanekuid ei ole.")\
            for _, row in df.iterrows():\
                sym = str(row["symbol"])\
                with st.container(border=True):\
                    a, b, c = st.columns([2, 2, 1])\
                    a.markdown(f"**\{sym\}**")\
                    a.write(f"Skoor: \{row['score']\}")\
                    a.write(f"Lead-lag boonus: \{row.get('leadlag_bonus', 0)\}")\
                    a.write(f"Confidence: \{row.get('confidence', 'NA')\} | \{row.get('conviction', 'Normal')\}")\
                    a.write(f"Hind: \{row['close']\}")\
                    b.write(f"Kogus: \{int(row['quantity'])\}")\
                    b.write(f"Eeldatav laekumine: \{fmt_money(float(row['estimated_cash_impact']))\}")\
                    b.write(f"P\'f5hjus: \{row['reason']\}")\
                    if c.button(f"Aksepteeri SELL \{sym\}", key=f"sell_\{sym\}", use_container_width=True):\
                        st.success(accept_sell(portfolio, sym, float(row['close']), str(row.get('reason', '')), None if pd.isna(row.get('score')) else float(row.get('score'))))\
                        st.rerun()\
\
        with hold_tab:\
            df = proposals[proposals["proposal_type"] == "HOLD"].sort_values("score", ascending=False)\
            if df.empty:\
                st.write("Avatud positsioonidele HOLD m\'e4rget t\'e4na ei ole.")\
            for _, row in df.iterrows():\
                sym = str(row["symbol"])\
                with st.container(border=True):\
                    a, b, c = st.columns([2, 2, 1])\
                    a.markdown(f"**\{sym\}**")\
                    a.write(f"Skoor: \{row['score']\}")\
                    a.write(f"Lead-lag boonus: \{row.get('leadlag_bonus', 0)\}")\
                    a.write(f"Confidence: \{row.get('confidence', 'NA')\} | \{row.get('conviction', 'Normal')\}")\
                    a.write(f"Hind: \{row['close']\}")\
                    b.write(f"Kogus portfellis: \{int(row['quantity'])\}")\
                    b.write(f"Stop viide: \{row.get('stop_reference')\}")\
                    b.write(f"P\'f5hjus: \{row['reason']\}")\
                    if c.button(f"M\'e4rgi HOLD \{sym\}", key=f"hold_\{sym\}", use_container_width=True):\
                        record_hold_review(sym, str(row.get('reason', '')), None if pd.isna(row.get('score')) else float(row.get('score')))\
                        st.success(f"HOLD \'fclevaatus salvestatud: \{sym\}.")\
\
    st.subheader("Portfelli seis")\
    portfolio_df = portfolio_snapshot_df(portfolio)\
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)\
    if not portfolio_df.empty:\
        st.caption(f"Avatud positsioonide realiseerimata P/L: \{fmt_money(float(portfolio_df['unrealized_pnl'].sum()))\}")\
\
    st.subheader("Lead-lag seosed")\
    if relationships.empty:\
        st.write("Lead-lag seoseid veel ei ole v\'f5i moodul on v\'e4lja l\'fclitatud.")\
    else:\
        st.dataframe(relationships.sort_values("corr", ascending=False), use_container_width=True, hide_index=True)\
\
    st.subheader("Suletud positsioonid")\
    closed = pd.DataFrame(portfolio.get("closed_positions", []))\
    if closed.empty:\
        st.write("Suletud positsioone veel ei ole.")\
    else:\
        cols = [c for c in ["symbol", "quantity", "avg_price", "exit_price", "realized_pnl", "opened_at", "closed_at", "close_reason"] if c in closed.columns]\
        st.dataframe(closed[cols].sort_values("closed_at", ascending=False), use_container_width=True, hide_index=True)\
\
    st.subheader("Tehingulogi")\
    if TRANSACTIONS_FILE.exists():\
        tx = pd.read_csv(TRANSACTIONS_FILE)\
        st.dataframe(tx.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)\
    else:\
        st.write("Tehinguid veel ei ole.")\
\
\
def run_scheduler(hour: int, minute: int, timezone: str) -> None:\
    tz = pytz.timezone(timezone)\
    scheduler = BlockingScheduler(timezone=tz)\
\
    # Final daily run (global close)\
    scheduler.add_job(\
        execute_daily_job,\
        "cron",\
        hour=hour,\
        minute=minute,\
        id="portfolio_daily_final",\
        replace_existing=True,\
        coalesce=True,\
        max_instances=1,\
    )\
\
    # Europe preview run (fixed 18:40 local time)\
    scheduler.add_job(\
        execute_daily_job,\
        "cron",\
        hour=18,\
        minute=40,\
        id="portfolio_daily_europe_preview",\
        replace_existing=True,\
        coalesce=True,\
        max_instances=1,\
    )\
\
    print(f"Scheduler started. Europe preview at 18:40 and final run at \{hour:02d\}:\{minute:02d\} (\{timezone\})")\
    scheduler.start()\
\
\
def parse_args() -> argparse.Namespace:\
    parser = argparse.ArgumentParser(description="Portfolio Signal Dashboard Lightyear")\
    parser.add_argument("--run-job", action="store_true", help="Run the daily scan job once")\
    parser.add_argument("--scheduler", action="store_true", help="Run daily scheduler")\
    parser.add_argument("--hour", type=int, default=18)\
    parser.add_argument("--minute", type=int, default=10)\
    parser.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE)\
    return parser.parse_args()\
\
\
def main() -> None:\
    args = parse_args()\
    if args.run_job:\
        result = execute_daily_job()\
        print(f"Daily job finished. Regime: \{result['regime']\}. Notifications: \{result['notification_message']\}")\
        return\
    if args.scheduler:\
        run_scheduler(args.hour, args.minute, args.timezone)\
        return\
    run_streamlit_app()\
\
\
if __name__ == "__main__":\
    main()\
}