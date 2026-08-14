# TradingView indicator and strategy review

This project was designed after reviewing TradingView's official built-in indicator catalog, built-in strategy catalog, Technical Ratings methodology, and Pine Script strategy/backtesting guidance.

## What was reviewed

TradingView's built-in catalog spans trend, momentum, volatility, volume/flow, price-channel, pivot, session, and composite tools. The official built-in strategy catalog includes:

- BarUpDn Strategy
- Bollinger Bands Strategy / Directed
- Channel BreakOut Strategy
- Consecutive Up/Down Strategy
- Greedy Strategy
- InSide Bar Strategy
- Keltner Channels Strategy
- MACD Strategy
- Momentum Strategy
- OutSide Bar Strategy
- Parabolic SAR Strategy
- Pivot Extension / Pivot Reversal
- Price Channel Strategy
- RSI Strategy
- Stochastic Slow Strategy
- Supertrend Strategy
- Technical Ratings Strategy
- Volty Expan Close Strategy

The official Technical Ratings indicator is especially useful as a reference because it deliberately aggregates 26 constituents instead of pretending one indicator is sufficient. Its moving-average group includes SMA and EMA 10/20/30/50/100/200, Hull MA, VWMA and Ichimoku. Its oscillator group includes RSI, Stochastic, CCI, ADX/DMI, Awesome Oscillator, Momentum, MACD, Stochastic RSI, Williams %R, Bull/Bear Power, and Ultimate Oscillator.

## Why the strategy does not simply require 30 indicators to agree

Many indicators are transforms of the same price history. Requiring all of them to agree can create a false feeling of confirmation while actually double-counting the same information and overfitting historical data.

Adaptive Confluence therefore groups signals by function:

1. **Trend**: SMA/EMA families, HMA, VWMA, Ichimoku, VWAP, Supertrend, higher-timeframe trend.
2. **Momentum**: RSI, MACD, CCI, Stochastic/Stoch RSI, Momentum, Williams %R, AO, Ultimate Oscillator, Bull/Bear Power.
3. **Volume/flow**: relative volume, MFI, CMF, OBV, VWMA-vs-SMA, VWAP.
4. **Volatility/regime**: ATR%, Bollinger width/%B, Keltner squeeze, ADX/DMI, Choppiness Index.
5. **Price structure**: Donchian/price channel, inside/outside bars, previous-day pivot.

Each group contributes a normalized score. Playbooks then use only the indicators relevant to their regime.

## Playbooks

### 1. Trend pullback / continuation

Used only when ADX and Choppiness indicate a directional market. Direction must agree with the ensemble, VWAP, EMA structure, Supertrend, and DMI. Entry waits for a controlled pullback toward EMA20 or VWAP, then continuation.

### 2. Squeeze / channel breakout

Combines Bollinger/Keltner compression, expansion, relative volume, MACD histogram, VWAP direction, and a Donchian/inside/outside-bar price break. This merges concepts from Bollinger, Keltner, Channel BreakOut, Price Channel, Momentum, and inside/outside-bar strategy families.

### 3. Range mean reversion

Only enabled when ADX is weak and CHOP is high. It fades a Bollinger extreme only after price begins moving back inside the band, with RSI, MFI, CMF and VWAP context. It is disabled in strong trends because Bollinger's own documentation warns that prices can “walk the bands.”

## Risk and realism rules

- Signals are evaluated on confirmed bars.
- Higher-timeframe values use TradingView's recommended non-repainting pattern: one-bar offset plus `barmerge.lookahead_on`.
- Pine strategy uses standard order simulation assumptions, slippage, commission, and Bar Magnifier support.
- Python backtester enters on the next bar's open after a signal, not the same close that created it.
- If stop and target are both touched in one bar, Python's default is conservative: stop first.
- Position size is based on stop distance and account risk, then capped by maximum notional exposure.
- Daily loss limits, trade-count limits, cooldowns, time stops, ATR stops, R targets, and trailing exits are included.
- Optimization script uses a chronological train/test split and keeps the final test section untouched.

## Official TradingView references

- Built-in Indicators: https://www.tradingview.com/support/folders/43000587405/
- Built-in Strategies: https://www.tradingview.com/support/folders/43000587406/
- Technical Ratings: https://www.tradingview.com/support/solutions/43000614331-technical-ratings/
- Strategies / Pine Script v6: https://www.tradingview.com/pine-script-docs/concepts/strategies/
- Repainting and higher timeframes: https://www.tradingview.com/pine-script-docs/concepts/repainting/
- VWAP: https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/
- RSI: https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
- ADX: https://www.tradingview.com/support/solutions/43000589099-average-directional-index-adx/
- MACD: https://www.tradingview.com/support/solutions/43000502344-moving-average-convergence-divergence-macd-indicator/
- Bollinger Bands: https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/
- Choppiness Index: https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/
- MFI: https://www.tradingview.com/support/solutions/43000502348-money-flow-mfi/
- OBV: https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/
- Supertrend: https://www.tradingview.com/support/solutions/43000634738-supertrend/

## Important limitation

This project reviews TradingView's official built-ins and strategy documentation. It does **not** scrape, copy, or claim to evaluate all 150,000+ Community Scripts. Those scripts include duplicated ideas, private/closed-source code, curve-fit systems, and content with varying quality. The goal here is a robust synthesis of the major independent indicator families, not indicator-count maximization for its own sake.
