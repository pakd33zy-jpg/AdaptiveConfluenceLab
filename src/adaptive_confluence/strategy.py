from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .indicators import atr, dmi_adx, ema, macd, rsi, session_vwap, sma, validate_ohlcv


@dataclass
class StrategyConfig:
    """V2.1 trend + breakout research configuration.

    V2.1 keeps explicit entry rules, but removes the excessive stacking that
    caused V2 to generate almost no trades on multi-year 5-minute data.
    Legacy V1 fields remain accepted for compatibility with older scripts.
    """

    score_threshold: float = 0.0
    range_adx_max: float = 19.0
    trend_chop_max: float = 55.0
    range_chop_min: float = 57.0

    trend_adx_min: float = 16.0
    min_relative_volume: float = 0.60
    breakout_relative_volume: float = 1.00
    min_atr_pct: float = 0.02
    max_atr_pct: float = 2.00
    donchian_length: int = 12
    session_start: str = "09:40"
    session_end: str = "15:45"

    # Conservative risk stays in place until an out-of-sample edge is proven.
    risk_pct: float = 0.25
    max_position_pct: float = 20.0
    trend_stop_atr: float = 1.20
    trend_target_r: float = 1.50
    breakout_stop_atr: float = 1.35
    breakout_target_r: float = 1.80
    mean_stop_atr: float = 1.00  # compatibility only; MEAN remains disabled
    mean_target_r: float = 1.35
    trail_activate_r: float = 1.00
    trail_atr: float = 1.25
    max_hold_bars: int = 30
    daily_loss_limit_pct: float = 1.00
    max_trades_per_day: int = 4
    cooldown_bars: int = 4
    allow_fractional: bool = True

    def to_dict(self):
        return asdict(self)


def _session_mask(index: pd.Index, start: str, end: str) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series(True, index=index, dtype=bool)
    local = index.tz_convert("America/New_York") if index.tz is not None else index.tz_localize("America/New_York")
    hhmm = local.hour * 60 + local.minute
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    return pd.Series((hhmm >= sh * 60 + sm) & (hhmm <= eh * 60 + em), index=index, dtype=bool)


def compute_features(raw: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """Build the V2.1 signal frame.

    The strategy uses two explicit playbooks:
      * TREND: pullback/reclaim around EMA20 in the direction of the broader bias.
      * BREAKOUT: prior-channel break with directional flow and usable volume.

    Signals are generated from a completed bar and filled by the backtester at
    the next bar open. Diagnostic boolean columns are intentionally retained so
    scripts/diagnose_signals.py can show where candidate bars are filtered out.
    """
    cfg = cfg or StrategyConfig()
    out = validate_ohlcv(raw).copy()

    out["ema20"] = ema(out.close, 20)
    out["ema50"] = ema(out.close, 50)
    out["ema200"] = ema(out.close, 200)
    out["vwap"] = session_vwap(out)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = 100.0 * out.atr14 / out.close.replace(0, np.nan)
    out["rsi14"] = rsi(out.close, 14)
    out["plus_di"], out["minus_di"], out["adx"] = dmi_adx(out, 14, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out.close)
    out["vol_sma20"] = sma(out.volume, 20)
    out["rel_vol"] = out.volume / out.vol_sma20.replace(0, np.nan)

    bar_range = (out.high - out.low).replace(0, np.nan)
    out["body_ratio"] = (out.close - out.open).abs() / bar_range
    out["ema20_distance_atr"] = (out.close - out.ema20).abs() / out.atr14.replace(0, np.nan)
    out["don_high"] = out.high.rolling(cfg.donchian_length, min_periods=cfg.donchian_length).max().shift(1)
    out["don_low"] = out.low.rolling(cfg.donchian_length, min_periods=cfg.donchian_length).min().shift(1)

    # V2 required EMA20 > EMA50 > EMA200 simultaneously. V2.1 keeps the
    # intermediate trend plus a broad price-vs-EMA200 bias, which is materially
    # less restrictive without abandoning trend context.
    out["long_trend_ok"] = (out.ema20 > out.ema50) & (out.close > out.ema200)
    out["short_trend_ok"] = (out.ema20 < out.ema50) & (out.close < out.ema200)
    out["session_ok"] = _session_mask(out.index, cfg.session_start, cfg.session_end)
    out["volatility_ok"] = out.atr_pct.between(cfg.min_atr_pct, cfg.max_atr_pct)
    out["adx_ok"] = out.adx >= cfg.trend_adx_min
    out["trend_volume_ok"] = out.rel_vol >= cfg.min_relative_volume
    out["breakout_volume_ok"] = out.rel_vol >= cfg.breakout_relative_volume
    out["long_vwap_ok"] = out.close > out.vwap
    out["short_vwap_ok"] = out.close < out.vwap
    out["long_di_ok"] = out.plus_di > out.minus_di
    out["short_di_ok"] = out.minus_di > out.plus_di

    score_parts = pd.concat(
        [
            pd.Series(np.where(out.ema20 > out.ema50, 1.0, -1.0), index=out.index),
            pd.Series(np.where(out.close > out.ema200, 1.0, -1.0), index=out.index),
            pd.Series(np.where(out.close > out.vwap, 1.0, -1.0), index=out.index),
            np.sign(out.plus_di - out.minus_di),
            np.sign(out.macd_hist),
        ],
        axis=1,
    )
    out["direction_score"] = score_parts.mean(axis=1)
    out["trend_regime"] = out.adx_ok & (out.long_trend_ok | out.short_trend_ok)
    out["range_regime"] = False
    out["chop"] = np.nan

    # Pullback touch + reclaim. VWAP and MACD are informative diagnostics but
    # are no longer both hard requirements. DI agreement supplies direction;
    # the signal bar only needs a modest confirmation candle.
    out["pull_touch_long"] = (
        (out.low <= out.ema20 + 0.15 * out.atr14) &
        (out.low >= out.ema50 - 0.60 * out.atr14)
    )
    out["pull_touch_short"] = (
        (out.high >= out.ema20 - 0.15 * out.atr14) &
        (out.high <= out.ema50 + 0.60 * out.atr14)
    )
    out["pull_confirm_long"] = (
        (out.close >= out.ema20) &
        ((out.close > out.open) | (out.close > out.close.shift(1))) &
        out.rsi14.between(42, 72) &
        (out.body_ratio >= 0.15)
    )
    out["pull_confirm_short"] = (
        (out.close <= out.ema20) &
        ((out.close < out.open) | (out.close < out.close.shift(1))) &
        out.rsi14.between(28, 58) &
        (out.body_ratio >= 0.15)
    )

    pull_long = (
        out.session_ok & out.volatility_ok & out.long_trend_ok & out.adx_ok &
        out.trend_volume_ok & out.long_di_ok & out.pull_touch_long & out.pull_confirm_long
    )
    pull_short = (
        out.session_ok & out.volatility_ok & out.short_trend_ok & out.adx_ok &
        out.trend_volume_ok & out.short_di_ok & out.pull_touch_short & out.pull_confirm_short
    )

    # Breakouts still require VWAP alignment and a prior-channel break, but the
    # volume/body/RSI gates are moderate enough to produce a testable sample.
    out["breakout_structure_long"] = (
        (out.close > out.don_high) & out.rsi14.between(50, 78) &
        (out.body_ratio >= 0.25) & (out.ema20_distance_atr <= 2.50)
    )
    out["breakout_structure_short"] = (
        (out.close < out.don_low) & out.rsi14.between(22, 50) &
        (out.body_ratio >= 0.25) & (out.ema20_distance_atr <= 2.50)
    )
    breakout_long = (
        out.session_ok & out.volatility_ok & out.long_trend_ok & out.adx_ok &
        out.breakout_volume_ok & out.long_vwap_ok & out.long_di_ok & out.breakout_structure_long
    )
    breakout_short = (
        out.session_ok & out.volatility_ok & out.short_trend_ok & out.adx_ok &
        out.breakout_volume_ok & out.short_vwap_ok & out.short_di_ok & out.breakout_structure_short
    )

    out["pull_long"] = pull_long
    out["pull_short"] = pull_short
    out["breakout_long"] = breakout_long
    out["breakout_short"] = breakout_short

    out["long_setup"] = np.select([breakout_long, pull_long], ["BREAKOUT", "TREND"], default="")
    out["short_setup"] = np.select([breakout_short, pull_short], ["BREAKOUT", "TREND"], default="")
    out["signal"] = np.select(
        [
            (out.long_setup != "") & (out.short_setup == ""),
            (out.short_setup != "") & (out.long_setup == ""),
        ],
        [1, -1],
        default=0,
    ).astype(int)
    out["setup"] = np.where(out.signal > 0, out.long_setup, np.where(out.signal < 0, out.short_setup, ""))
    return out


def setup_risk(cfg: StrategyConfig, setup: str):
    if setup == "BREAKOUT":
        return cfg.breakout_stop_atr, cfg.breakout_target_r
    if setup == "MEAN":
        return cfg.mean_stop_atr, cfg.mean_target_r
    return cfg.trend_stop_atr, cfg.trend_target_r
