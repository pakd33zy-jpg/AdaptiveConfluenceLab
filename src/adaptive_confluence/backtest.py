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


def _bucket_stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "net_pnl": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
    wins = frame.loc[frame.net_pnl > 0, "net_pnl"]
    losses = frame.loc[frame.net_pnl < 0, "net_pnl"]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "trades": int(len(frame)),
        "net_pnl": float(frame.net_pnl.sum()),
        "win_rate_pct": float(100 * (frame.net_pnl > 0).mean()),
        "profit_factor": float(pf),
        "expectancy": float(frame.net_pnl.mean()),
    }


def run_backtest(
    raw: pd.DataFrame,
    strategy_cfg: StrategyConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
    *,
    precomputed_features: pd.DataFrame | None = None,
) -> BacktestResult:
    sc = strategy_cfg or StrategyConfig()
    bc = bt_cfg or BacktestConfig()
    df = precomputed_features if precomputed_features is not None else compute_features(raw, sc)

    # Pull frequently accessed columns into NumPy arrays. V1 used df.iloc in the
    # inner loop, which dominated the runtime on multi-year 5-minute data.
    idx = df.index
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    atrs = df["atr14"].to_numpy(float)
    signals = df["signal"].to_numpy(int)
    setups = df["setup"].astype(str).to_numpy()
    scores = df["direction_score"].to_numpy(float)

    cash = bc.initial_capital
    equity = bc.initial_capital
    position = None
    trades = []
    eq_times = []
    eq_values = []
    daily_start_equity = equity
    current_day = None
    trades_today = 0
    last_entry_i = -10_000

    for i in range(1, len(df)):
        ts = idx[i]
        day = ts.date() if hasattr(ts, "date") else None
        if day != current_day:
            current_day = day
            daily_start_equity = equity
            trades_today = 0

        if position is not None:
            d = position["direction"]
            equity = cash + d * position["qty"] * (closes[i] - position["entry_price"])
        else:
            equity = cash

        day_pnl_pct = 100 * (equity - daily_start_equity) / daily_start_equity if daily_start_equity else 0.0

        if position is not None:
            d = position["direction"]
            position["best_high"] = max(position["best_high"], highs[i])
            position["best_low"] = min(position["best_low"], lows[i])
            entry = position["entry_price"]
            stop_dist = position["stop_dist"]
            target_r = position["target_r"]
            age = i - position["entry_i"]

            if d > 0:
                initial_stop = entry - stop_dist
                target = entry + stop_dist * target_r
                if position["best_high"] >= entry + stop_dist * sc.trail_activate_r:
                    trail = position["best_high"] - atrs[i] * sc.trail_atr
                    stop = max(initial_stop, entry, trail)
                else:
                    stop = initial_stop
                hit_stop = lows[i] <= stop
                hit_target = highs[i] >= target
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
                    exit_reason, exit_price = "TIME", closes[i]
            else:
                initial_stop = entry + stop_dist
                target = entry - stop_dist * target_r
                if position["best_low"] <= entry - stop_dist * sc.trail_activate_r:
                    trail = position["best_low"] + atrs[i] * sc.trail_atr
                    stop = min(initial_stop, entry, trail)
                else:
                    stop = initial_stop
                hit_stop = highs[i] >= stop
                hit_target = lows[i] <= target
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
                    exit_reason, exit_price = "TIME", closes[i]

            if exit_price is not None:
                px = _slip(float(exit_price), -d, bc.slippage_bps)
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
                    "r_multiple": net / position["risk_cash"] if position["risk_cash"] else 0.0,
                })
                position = None
                equity = cash

        # Signal is from previous completed close; fill is current open.
        if position is None and signals[i - 1] != 0:
            if (
                trades_today < sc.max_trades_per_day
                and day_pnl_pct > -sc.daily_loss_limit_pct
                and (i - last_entry_i) > sc.cooldown_bars
            ):
                d = int(signals[i - 1])
                setup = str(setups[i - 1])
                stop_mult, target_r = setup_risk(sc, setup)
                atr_val = float(atrs[i - 1])
                if math.isfinite(atr_val) and atr_val > 0:
                    entry = _slip(float(opens[i]), d, bc.slippage_bps)
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
                            "best_high": float(highs[i]),
                            "best_low": float(lows[i]),
                            "score": float(scores[i - 1]),
                        }
                        trades_today += 1
                        last_entry_i = i

        eq_times.append(ts)
        eq_values.append(equity)

    # Realize any position still open at the end of the dataset so the final
    # equity and trade statistics do not silently ignore an unfinished trade.
    if position is not None and len(df):
        d = position["direction"]
        px = _slip(float(closes[-1]), -d, bc.slippage_bps)
        gross = d * position["qty"] * (px - position["entry_price"])
        exit_commission = abs(position["qty"] * px) * bc.commission_pct / 100
        net = gross - position["entry_commission"] - exit_commission
        cash += net
        trades.append({
            **position,
            "exit_time": idx[-1],
            "exit_price": px,
            "exit_reason": "END_OF_DATA",
            "gross_pnl": gross,
            "net_pnl": net,
            "r_multiple": net / position["risk_cash"] if position["risk_cash"] else 0.0,
        })
        equity = cash
        if eq_values:
            eq_values[-1] = cash

    eq = pd.Series(eq_values, index=eq_times, dtype=float).sort_index()
    ret = eq.pct_change().fillna(0)
    peak = eq.cummax()
    dd = (eq / peak - 1) * 100
    trade_df = pd.DataFrame(trades)
    net_pnl = cash - bc.initial_capital

    base = _bucket_stats(trade_df)
    sharpe = np.sqrt(252 * 78) * ret.mean() / ret.std(ddof=0) if ret.std(ddof=0) > 0 else 0.0

    by_setup = {}
    by_direction = {}
    by_exit_reason = {}
    if not trade_df.empty:
        for name, group in trade_df.groupby("setup"):
            by_setup[str(name)] = _bucket_stats(group)
        for d, group in trade_df.groupby("direction"):
            by_direction["LONG" if int(d) > 0 else "SHORT"] = _bucket_stats(group)
        for reason, group in trade_df.groupby("exit_reason"):
            by_exit_reason[str(reason)] = _bucket_stats(group)

    stats = {
        "initial_capital": bc.initial_capital,
        "ending_equity": cash,
        "net_pnl": net_pnl,
        "return_pct": 100 * net_pnl / bc.initial_capital,
        "trades": base["trades"],
        "wins": int((trade_df.net_pnl > 0).sum()) if not trade_df.empty else 0,
        "losses": int((trade_df.net_pnl < 0).sum()) if not trade_df.empty else 0,
        "win_rate_pct": base["win_rate_pct"],
        "profit_factor": base["profit_factor"],
        "expectancy": base["expectancy"],
        "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "sharpe_approx": float(sharpe),
        "by_setup": by_setup,
        "by_direction": by_direction,
        "by_exit_reason": by_exit_reason,
    }
    return BacktestResult(stats=stats, trades=trade_df, equity_curve=eq, features=df)
