from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from .strategy import StrategyConfig, compute_features, setup_risk


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.01
    slippage_bps: float = 1.0
    conservative_same_bar: bool = True


class BacktestResult(dict):
    pass


def _slip(price: float, side: int, bps: float) -> float:
    return price * (1 + side * bps / 10_000.0)


def run_backtest(raw: pd.DataFrame, strategy_cfg: StrategyConfig | None = None, bt_cfg: BacktestConfig | None = None) -> BacktestResult:
    sc = strategy_cfg or StrategyConfig()
    bc = bt_cfg or BacktestConfig()
    df = compute_features(raw, sc).copy()

    cash = bc.initial_capital
    equity = bc.initial_capital
    position = None
    trades = []
    equity_curve = []
    daily_start_equity = equity
    current_day = None
    trades_today = 0
    last_entry_i = -10_000

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = df.index[i]
        day = ts.date() if hasattr(ts, "date") else None
        if day != current_day:
            current_day = day
            daily_start_equity = equity
            trades_today = 0

        # Mark-to-market.
        if position:
            direction = position["direction"]
            equity = cash + direction * position["qty"] * (row.close - position["entry_price"])
        else:
            equity = cash

        day_pnl_pct = 100 * (equity - daily_start_equity) / daily_start_equity if daily_start_equity else 0

        # Exit management. Use current OHLC; if stop and target both touch in one
        # bar, conservative mode assumes the stop happened first.
        if position:
            d = position["direction"]
            position["best_high"] = max(position["best_high"], row.high)
            position["best_low"] = min(position["best_low"], row.low)
            entry = position["entry_price"]
            stop_dist = position["stop_dist"]
            target_r = position["target_r"]
            age = i - position["entry_i"]

            if d > 0:
                initial_stop = entry - stop_dist
                target = entry + stop_dist * target_r
                if position["best_high"] >= entry + stop_dist * sc.trail_activate_r:
                    trail = position["best_high"] - row.atr14 * sc.trail_atr
                    stop = max(initial_stop, entry, trail)
                else:
                    stop = initial_stop
                hit_stop = row.low <= stop
                hit_target = row.high >= target
                exit_reason = None
                exit_price = None
                if hit_stop and hit_target:
                    exit_reason = "STOP_SAME_BAR" if bc.conservative_same_bar else "TARGET_SAME_BAR"
                    exit_price = stop if bc.conservative_same_bar else target
                elif hit_stop:
                    exit_reason, exit_price = "STOP", stop
                elif hit_target:
                    exit_reason, exit_price = "TARGET", target
                elif age >= sc.max_hold_bars:
                    exit_reason, exit_price = "TIME", row.close
            else:
                initial_stop = entry + stop_dist
                target = entry - stop_dist * target_r
                if position["best_low"] <= entry - stop_dist * sc.trail_activate_r:
                    trail = position["best_low"] + row.atr14 * sc.trail_atr
                    stop = min(initial_stop, entry, trail)
                else:
                    stop = initial_stop
                hit_stop = row.high >= stop
                hit_target = row.low <= target
                exit_reason = None
                exit_price = None
                if hit_stop and hit_target:
                    exit_reason = "STOP_SAME_BAR" if bc.conservative_same_bar else "TARGET_SAME_BAR"
                    exit_price = stop if bc.conservative_same_bar else target
                elif hit_stop:
                    exit_reason, exit_price = "STOP", stop
                elif hit_target:
                    exit_reason, exit_price = "TARGET", target
                elif age >= sc.max_hold_bars:
                    exit_reason, exit_price = "TIME", row.close

            if exit_price is not None:
                px = _slip(exit_price, -d, bc.slippage_bps)
                gross = d * position["qty"] * (px - entry)
                exit_commission = abs(position["qty"] * px) * bc.commission_pct / 100
                net = gross - position["entry_commission"] - exit_commission
                cash += net
                trades.append({
                    **position,
                    "exit_time": ts,
                    "exit_price": px,
                    "exit_reason": exit_reason,
                    "gross_pnl": gross,
                    "net_pnl": net,
                    "r_multiple": net / position["risk_cash"] if position["risk_cash"] else 0,
                })
                position = None
                equity = cash

        # Signal from previous close -> current open entry. This avoids using the
        # close that generated a signal as if we already knew it intrabar.
        if position is None and prev.signal != 0:
            if trades_today < sc.max_trades_per_day and day_pnl_pct > -sc.daily_loss_limit_pct and (i - last_entry_i) > 3:
                d = int(prev.signal)
                setup = str(prev.setup)
                stop_mult, target_r = setup_risk(sc, setup)
                atr_val = float(prev.atr14)
                if math.isfinite(atr_val) and atr_val > 0:
                    entry = _slip(float(row.open), d, bc.slippage_bps)
                    stop_dist = atr_val * stop_mult
                    risk_cash = equity * sc.risk_pct / 100
                    risk_qty = risk_cash / stop_dist
                    cap_qty = equity * sc.max_position_pct / 100 / entry
                    qty = min(risk_qty, cap_qty)
                    if not sc.allow_fractional:
                        qty = math.floor(qty)
                    if qty > 0:
                        entry_commission = abs(qty * entry) * bc.commission_pct / 100
                        position = {
                            "entry_time": ts,
                            "entry_i": i,
                            "direction": d,
                            "setup": setup,
                            "entry_price": entry,
                            "qty": qty,
                            "stop_dist": stop_dist,
                            "target_r": target_r,
                            "risk_cash": risk_cash,
                            "entry_commission": entry_commission,
                            "best_high": float(row.high),
                            "best_low": float(row.low),
                            "score": float(prev.direction_score),
                        }
                        trades_today += 1
                        last_entry_i = i

        equity_curve.append((ts, equity))

    eq = pd.Series({t: e for t, e in equity_curve}).sort_index()
    ret = eq.pct_change().fillna(0)
    peak = eq.cummax()
    dd = (eq / peak - 1) * 100
    trade_df = pd.DataFrame(trades)
    net_pnl = cash - bc.initial_capital
    wins = trade_df[trade_df.net_pnl > 0] if not trade_df.empty else trade_df
    losses = trade_df[trade_df.net_pnl < 0] if not trade_df.empty else trade_df
    gross_win = wins.net_pnl.sum() if not trade_df.empty else 0.0
    gross_loss = -losses.net_pnl.sum() if not trade_df.empty else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    sharpe = np.sqrt(252 * 78) * ret.mean() / ret.std(ddof=0) if ret.std(ddof=0) > 0 else 0.0

    stats = {
        "initial_capital": bc.initial_capital,
        "ending_equity": cash,
        "net_pnl": net_pnl,
        "return_pct": 100 * net_pnl / bc.initial_capital,
        "trades": int(len(trade_df)),
        "wins": int((trade_df.net_pnl > 0).sum()) if not trade_df.empty else 0,
        "losses": int((trade_df.net_pnl < 0).sum()) if not trade_df.empty else 0,
        "win_rate_pct": float(100 * (trade_df.net_pnl > 0).mean()) if not trade_df.empty else 0.0,
        "profit_factor": float(pf),
        "expectancy": float(trade_df.net_pnl.mean()) if not trade_df.empty else 0.0,
        "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "sharpe_approx": float(sharpe),
    }
    return BacktestResult(stats=stats, trades=trade_df, equity_curve=eq, features=df)
