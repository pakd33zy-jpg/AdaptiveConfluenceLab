# Validation plan before any live deployment

## Stage 1 — TradingView sanity checks

Use standard candles, realistic strategy properties, and Bar Magnifier where available. Confirm that signals do not disappear after reload. Verify higher-timeframe inputs are higher than the chart timeframe.

## Stage 2 — Broad historical robustness

Do not optimize on a single ticker. Test a broad liquid universe across:

- index ETFs
- mega-cap technology
- financials
- industrials
- healthcare
- energy
- high-volatility and low-volatility stocks

Track return, max drawdown, profit factor, expectancy per trade, trade count, and performance by playbook.

## Stage 3 — Walk-forward

Use chronological windows. Optimize only a small number of high-level parameters on the training window, then freeze them for the next untouched window. Roll forward and repeat.

Avoid giant parameter searches. If profitability disappears when a parameter moves slightly, the strategy is fragile.

## Stage 4 — Cost stress

Repeat with at least 2x expected slippage/fees. A strategy that only survives perfect fills is not robust.

## Stage 5 — Paper forward test

Run at least 30-50 trades per playbook before drawing conclusions. Prefer 100+ total trades over varied conditions. Compare actual paper fills with TradingView assumptions.

## Stage 6 — Deployment gate

Only consider live execution if:

- out-of-sample expectancy is positive,
- drawdown is acceptable,
- profit does not come from one symbol or one month,
- multiple playbooks or regimes contribute,
- actual paper fills are close to the modeled costs,
- no lookahead/repainting artifacts are present.
