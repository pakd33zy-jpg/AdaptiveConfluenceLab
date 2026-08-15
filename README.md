# AdaptiveConfluenceLab

A research repository for a regime-adaptive TradingView/Python trading strategy built from the major independent technical-analysis families rather than one “magic” indicator.

## What is inside

- `pine/adaptive_confluence_strategy.pine` — Pine Script v6 TradingView strategy.
- `src/adaptive_confluence/` — pure pandas/numpy indicator, scoring, and backtest engine.
- `scripts/fetch_alpaca.py` — fetch OHLCV bars from Alpaca Data API.
- `scripts/backtest_csv.py` — run a realistic next-bar backtest on CSV data.
- `scripts/walk_forward.py` — chronological train/test robustness sweep.
- `research/tradingview_review.md` — indicator/strategy research notes and official references.
- `tests/` — deterministic tests for core calculations and smoke tests.

## Strategy design

The system uses 30+ indicator components but avoids treating correlated indicators as independent evidence. It groups them into trend, momentum, volume/flow, volatility/regime, and price-structure scores.

Three adaptive playbooks are enabled:

1. **TREND** — VWAP/EMA pullback continuation in strong directional regimes.
2. **BREAKOUT** — Bollinger/Keltner squeeze or channel breakout with relative-volume expansion.
3. **MEAN** — Bollinger/VWAP mean reversion only in range regimes.

Risk is stop-distance based, not “all-in.” The Pine and Python implementations include ATR stops, R targets, trailing protection, time stops, max trades/day, daily loss limits, and position-notional caps.

## TradingView use

1. Open TradingView Pine Editor.
2. Paste `pine/adaptive_confluence_strategy.pine`.
3. Add it to a **standard candlestick chart**.
4. Start with liquid US equities on 1m/5m charts; set confirmed HTF above the chart timeframe (e.g. 15m on a 5m chart).
5. In Strategy Tester, use realistic commission/slippage and enable Bar Magnifier if your plan supports it.
6. Test many symbols and market regimes. Do not judge the strategy from one ticker or one month.

## Python setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### Fetch Alpaca bars

```bash
export ALPACA_PAPER_API_KEY="..."
export ALPACA_PAPER_SECRET_KEY="..."
python scripts/fetch_alpaca.py SPY --start 2026-01-01T14:30:00Z --end 2026-08-01T20:00:00Z --timeframe 5Min
```

### Backtest

```bash
python scripts/backtest_csv.py SPY_5Min.csv --capital 100000 --commission-pct 0.01 --slippage-bps 1
```

### Walk-forward check

```bash
python scripts/walk_forward.py SPY_5Min.csv
```

The walk-forward script intentionally leaves the final 30% untouched while selecting parameters on the first 70%. Repeat this across multiple symbols and different date ranges.

## Recommended research universe

For an equity strategy, test at minimum:

- SPY, QQQ, IWM
- AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA
- several financial, industrial, healthcare, and energy names
- high-volatility and low-volatility months
- strong bull, bear, and sideways regimes

A strategy that only works on one symbol or one hand-picked period is not ready.

## What “best” means here

There is no indicator combination that can be guaranteed to be the “most profitable.” The practical goal is **positive out-of-sample expectancy after realistic costs with tolerable drawdown**. The project is built to make that measurable instead of assuming a high backtest return means the system is good.

## Safety

This repository is for research and paper testing. Do not connect it to live order execution until you have enough out-of-sample and forward-test evidence to justify the risk.

## V26 Alpaca paper runner

`V26` is the frozen ETF momentum-rotation candidate selected during the research loop.

Rules are intentionally hard-coded in `scripts/paper_v26.py` so forward results do
not silently retune the backtest:

- Universe: SPY, QQQ, IWM, DIA, XLK, XLF, XLE, GLD, TLT
- 63 completed trading-session momentum
- Eligible only when momentum is positive and price is above SMA150
- Rank eligible ETFs by momentum
- Rebalance after every 5 completed trading sessions
- 70% to rank #1 and 30% to rank #2
- If only one ETF qualifies, the remaining 30% stays cash
- Alpaca **paper endpoint is hard-coded**; this runner does not submit live orders

Set paper credentials in the current shell:

```powershell
$env:ALPACA_PAPER_API_KEY="YOUR_PAPER_KEY"
$env:ALPACA_PAPER_SECRET_KEY="YOUR_PAPER_SECRET"
```

Preview the next V26 action without sending an order:

```powershell
python .\scripts\paper_v26.py
```

Submit the proposed orders to Alpaca **paper trading**:

```powershell
python .\scripts\paper_v26.py --execute
```

The first execution refuses to touch a non-empty paper account by default. This
prevents V26 from accidentally interfering with another paper strategy. Use a clean
paper account/reset when possible. Runtime state and event logs are written under
`paper/` and are ignored by Git.
