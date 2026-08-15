from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .indicators import atr, dmi_adx, ema, macd, rsi, session_vwap, sma, validate_ohlcv


@dataclass
class StrategyConfig:
    # V2 intentionally uses a smaller set of independent conditions instead of
    # averaging dozens of correlated indicators into a single score. Legacy V1
    # fields remain accepted so older scripts/configs do not break; they are not
    # used by the V2 entry rules unless noted below.
    score_threshold: float = 0.0
    range_adx_max: float = 19.0
    trend_chop_max: float = 55.0
    range_chop_min: float = 57.0
    trend_adx_min: float = 22.0
    min_relative_volume: float = 0.90
    breakout_relative_volume: float = 1.25
    min_atr_pct: float = 0.05
    max_atr_pct: float = 1.50
    donchian_length: int = 20
    session_start: str = "09:45"
    session_end: str = "15:30"

    # Risk defaults are deliberately smaller than V1 while the edge is being
    # validated. Position size is still capped by both stop risk and notional.
    risk_pct: float = 0.25
    max_position_pct: float = 20.0
    trend_stop_atr: float = 1.10
    trend_target_r: float = 1.80
    breakout_stop_atr: float = 1.30
    breakout_target_r: float = 2.20
    mean_stop_atr: float = 1.00  # retained for API compatibility; V2 does not trade MEAN
    mean_target_r: float = 1.35
    trail_activate_r: float = 1.00
    trail_atr: float = 1.20
    max_hold_bars: int = 24
    daily_loss_limit_pct: float = 1.00
    max_trades_per_day: int = 4
    cooldown_bars: int = 6
    allow_fractional: bool = True

    def to_dict(self):
        return asdict(self)


def _session_mask(index: pd.Index, start: str, end: str) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series(True, index=index)
    local = index.tz_convert("America/New_York") if index.tz is not None else index.tz_localize("America/New_York")
    hhmm = local.hour * 60 + local.minute
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    return pd.Series((hhmm >= sh * 60 + sm) & (hhmm <= eh * 60 + em), index=index)


def compute_features(raw: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """Build the V2 signal frame.

    V2 is intentionally simpler than V1. It trades only two playbooks:
      * TREND: pullback/reclaim in an established EMA/VWAP trend.
      * BREAKOUT: Donchian break with volume/momentum expansion.

    Signals are generated from the completed bar and the backtester enters on
    the next bar open, so no current-bar close is used as a fill price.
    """
    cfg = cfg or StrategyConfig()
    out = validate_ohlcv(raw).copy()

    # Core trend / momentum / liquidity features.
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

    # Basic candle quality. This avoids counting tiny indecision bars as
    # confirmations and avoids breakouts that are already extremely stretched.
    bar_range = (out.high - out.low).replace(0, np.nan)
    out["body_ratio"] = (out.close - out.open).abs() / bar_range
    out["ema20_distance_atr"] = (out.close - out.ema20).abs() / out.atr14.replace(0, np.nan)

    # Price structure uses only prior bars for channel levels.
    out["don_high"] = out.high.rolling(cfg.donchian_length, min_periods=cfg.donchian_length).max().shift(1)
    out["don_low"] = out.low.rolling(cfg.donchian_length, min_periods=cfg.donchian_length).min().shift(1)

    long_trend = (out.ema20 > out.ema50) & (out.ema50 > out.ema200)
    short_trend = (out.ema20 < out.ema50) & (out.ema50 < out.ema200)
    long_flow = (out.close > out.vwap) & (out.plus_di > out.minus_di) & (out.macd_hist > 0)
    short_flow = (out.close < out.vwap) & (out.minus_di > out.plus_di) & (out.macd_hist < 0)
    volatility_ok = out.atr_pct.between(cfg.min_atr_pct, cfg.max_atr_pct)
    session_ok = _session_mask(out.index, cfg.session_start, cfg.session_end)

    # A compact diagnostic score retained for reporting/compatibility. It is not
    # used as a magic threshold; every entry rule below remains explicit.
    score_parts = pd.concat(
        [
            pd.Series(np.where(out.ema20 > out.ema50, 1.0, -1.0), index=out.index),
            pd.Series(np.where(out.ema50 > out.ema200, 1.0, -1.0), index=out.index),
            pd.Series(np.where(out.close > out.vwap, 1.0, -1.0), index=out.index),
            np.sign(out.plus_di - out.minus_di),
            np.sign(out.macd_hist),
        ],
        axis=1,
    )
    out["direction_score"] = score_parts.mean(axis=1)
    out["trend_regime"] = (out.adx >= cfg.trend_adx_min) & (long_trend | short_trend)
    out["range_regime"] = False  # V2 deliberately disables mean reversion.
    out["volatility_ok"] = volatility_ok
    out["chop"] = np.nan  # compatibility with existing smoke tests/reports

    # Trend pullback/reclaim. Require the signal bar to touch the fast trend
    # reference and close back in the trend direction with momentum improving.
    pull_long = (
        session_ok & volatility_ok & long_trend & long_flow &
        (out.adx >= cfg.trend_adx_min) &
        (out.rel_vol >= cfg.min_relative_volume) &
        ((out.low <= out.ema20) | (out.low <= out.vwap)) &
        (out.close > out.ema20) &
        (out.close > out.open) & (out.close > out.close.shift(1)) &
        out.rsi14.between(50, 68) &
        (out.macd_hist >= out.macd_hist.shift(1)) &
        (out.body_ratio >= 0.25)
    )
    pull_short = (
        session_ok & volatility_ok & short_trend & short_flow &
        (out.adx >= cfg.trend_adx_min) &
        (out.rel_vol >= cfg.min_relative_volume) &
        ((out.high >= out.ema20) | (out.high >= out.vwap)) &
        (out.close < out.ema20) &
        (out.close < out.open) & (out.close < out.close.shift(1)) &
        out.rsi14.between(32, 50) &
        (out.macd_hist <= out.macd_hist.shift(1)) &
        (out.body_ratio >= 0.25)
    )

    # Breakout continuation. A real channel break is required; V1's inside /
    # outside-bar alternatives were too permissive and created many weak trades.
    breakout_long = (
        session_ok & volatility_ok & long_trend & long_flow &
        (out.adx >= cfg.trend_adx_min) &
        (out.rel_vol >= cfg.breakout_relative_volume) &
        (out.close > out.don_high) &
        out.rsi14.between(55, 75) &
        (out.body_ratio >= 0.45) &
        (out.ema20_distance_atr <= 2.0)
    )
    breakout_short = (
        session_ok & volatility_ok & short_trend & short_flow &
        (out.adx >= cfg.trend_adx_min) &
        (out.rel_vol >= cfg.breakout_relative_volume) &
        (out.close < out.don_low) &
        out.rsi14.between(25, 45) &
        (out.body_ratio >= 0.45) &
        (out.ema20_distance_atr <= 2.0)
    )

    # BREAKOUT gets priority if a bar also qualifies as a pullback continuation.
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
