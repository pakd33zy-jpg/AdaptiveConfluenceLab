from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .indicators import (
    atr, bollinger, cci, choppiness, cmf, dmi_adx, ema, hma, macd,
    mfi, obv, roc, rsi, session_vwap, sma, stochastic, supertrend,
    ultimate_oscillator, validate_ohlcv, vwma, williams_r,
)


@dataclass
class StrategyConfig:
    score_threshold: float = 0.28
    trend_adx_min: float = 22.0
    range_adx_max: float = 19.0
    trend_chop_max: float = 55.0
    range_chop_min: float = 57.0
    min_relative_volume: float = 0.85
    breakout_relative_volume: float = 1.15
    max_atr_pct: float = 2.50
    donchian_length: int = 20
    risk_pct: float = 0.50
    max_position_pct: float = 25.0
    trend_stop_atr: float = 1.20
    trend_target_r: float = 2.0
    breakout_stop_atr: float = 1.35
    breakout_target_r: float = 2.40
    mean_stop_atr: float = 1.0
    mean_target_r: float = 1.35
    trail_activate_r: float = 1.0
    trail_atr: float = 1.10
    max_hold_bars: int = 36
    daily_loss_limit_pct: float = 2.0
    max_trades_per_day: int = 8
    allow_fractional: bool = True

    def to_dict(self):
        return asdict(self)


def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x).astype(float)


def _vote(buy: pd.Series, sell: pd.Series) -> pd.Series:
    return pd.Series(np.select([buy, sell], [1.0, -1.0], default=0.0), index=buy.index)


def compute_features(raw: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StrategyConfig()
    df = validate_ohlcv(raw)
    out = df.copy()

    # Moving-average family, mirroring the broad groups used by TradingView's
    # Technical Ratings rather than relying on a single crossover.
    sma_lengths = [10, 20, 30, 50, 100, 200]
    ema_lengths = [10, 20, 30, 50, 100, 200]
    ma_votes = []
    for n in sma_lengths:
        out[f"sma{n}"] = sma(out.close, n)
        ma_votes.append(_sign(out.close - out[f"sma{n}"]))
    for n in ema_lengths:
        out[f"ema{n}"] = ema(out.close, n)
        ma_votes.append(_sign(out.close - out[f"ema{n}"]))
    out["hma9"] = hma(out.close, 9)
    out["vwma20"] = vwma(out.close, out.volume, 20)
    ma_votes.extend([_sign(out.close - out.hma9), _sign(out.close - out.vwma20)])

    tenkan = (out.high.rolling(9).max() + out.low.rolling(9).min()) / 2
    kijun = (out.high.rolling(26).max() + out.low.rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (out.high.rolling(52).max() + out.low.rolling(52).min()) / 2
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    ichi = _vote((out.close > cloud_top) & (tenkan > kijun) & (span_a > span_b),
                 (out.close < cloud_bottom) & (tenkan < kijun) & (span_a < span_b))
    ma_votes.append(ichi)
    out["ma_rating"] = pd.concat(ma_votes, axis=1).mean(axis=1)

    # Oscillator family.
    out["rsi14"] = rsi(out.close, 14)
    out["stoch_k"], out["stoch_d"] = stochastic(out, 14, 3, 3)
    out["cci20"] = cci(out, 20)
    out["plus_di"], out["minus_di"], out["adx"] = dmi_adx(out, 14, 14)
    out["ao"] = sma((out.high + out.low) / 2, 5) - sma((out.high + out.low) / 2, 34)
    out["momentum10"] = out.close - out.close.shift(10)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out.close)
    rsi_low = out.rsi14.rolling(14).min()
    rsi_high = out.rsi14.rolling(14).max()
    stoch_rsi_raw = 100 * (out.rsi14 - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    out["stoch_rsi_k"] = sma(stoch_rsi_raw, 3)
    out["stoch_rsi_d"] = sma(out.stoch_rsi_k, 3)
    out["willr"] = williams_r(out, 14)
    out["ema13"] = ema(out.close, 13)
    out["bull_power"] = out.high - out.ema13
    out["bear_power"] = out.low - out.ema13
    out["uo"] = ultimate_oscillator(out)

    rsi_vote = _vote((out.rsi14 < 30) & (out.rsi14 > out.rsi14.shift(1)),
                     (out.rsi14 > 70) & (out.rsi14 < out.rsi14.shift(1)))
    stoch_vote = _vote((out.stoch_k < 20) & (out.stoch_d < 20) & (out.stoch_k > out.stoch_d),
                       (out.stoch_k > 80) & (out.stoch_d > 80) & (out.stoch_k < out.stoch_d))
    cci_vote = _vote((out.cci20 < -100) & (out.cci20 > out.cci20.shift(1)),
                     (out.cci20 > 100) & (out.cci20 < out.cci20.shift(1)))
    dmi_vote = _vote((out.plus_di > out.minus_di) & (out.adx > 20) & (out.adx >= out.adx.shift(1)),
                     (out.minus_di > out.plus_di) & (out.adx > 20) & (out.adx >= out.adx.shift(1)))
    ao_vote = _sign(out.ao)
    mom_vote = _sign(out.momentum10.diff())
    macd_vote = _sign(out.macd - out.macd_signal)
    srsi_vote = _vote((out.stoch_rsi_k < 20) & (out.stoch_rsi_d < 20) & (out.stoch_rsi_k > out.stoch_rsi_d),
                      (out.stoch_rsi_k > 80) & (out.stoch_rsi_d > 80) & (out.stoch_rsi_k < out.stoch_rsi_d))
    will_vote = _vote((out.willr < -80) & (out.willr > out.willr.shift(1)),
                      (out.willr > -20) & (out.willr < out.willr.shift(1)))
    bbp_vote = _vote((out.close > out.ema13) & (out.bear_power < 0) & (out.bear_power > out.bear_power.shift(1)),
                     (out.close < out.ema13) & (out.bull_power > 0) & (out.bull_power < out.bull_power.shift(1)))
    uo_vote = _vote((out.uo > 55) & (out.uo > out.uo.shift(1)), (out.uo < 45) & (out.uo < out.uo.shift(1)))
    out["osc_rating"] = pd.concat(
        [rsi_vote, stoch_vote, cci_vote, dmi_vote, ao_vote, mom_vote, macd_vote,
         srsi_vote, will_vote, bbp_vote, uo_vote], axis=1
    ).mean(axis=1)

    # Volume / flow.
    out["vwap"] = session_vwap(out)
    out["mfi14"] = mfi(out, 14)
    out["cmf20"] = cmf(out, 20)
    out["obv"] = obv(out)
    out["obv_ema20"] = ema(out.obv, 20)
    out["rel_vol"] = out.volume / sma(out.volume, 20).replace(0, np.nan)
    out["volume_score"] = pd.concat([
        _sign(out.close - out.vwap), _sign(out.mfi14 - 50), _sign(out.cmf20),
        _sign(out.obv - out.obv_ema20), _sign(out.vwma20 - out.sma20),
        _sign(out.close - out.open) * (out.rel_vol / 1.5).clip(0, 1),
    ], axis=1).mean(axis=1)

    # Volatility / regime.
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = 100 * out.atr14 / out.close.replace(0, np.nan)
    out["bb_basis"], out["bb_upper"], out["bb_lower"], out["bb_width"], out["bb_pctb"] = bollinger(out.close)
    out["bb_width_avg"] = sma(out.bb_width, 50)
    out["kc_basis"] = ema(out.close, 20)
    out["kc_upper"] = out.kc_basis + 1.5 * out.atr14
    out["kc_lower"] = out.kc_basis - 1.5 * out.atr14
    out["squeeze_on"] = (out.bb_upper < out.kc_upper) & (out.bb_lower > out.kc_lower)
    out["squeeze_release"] = out.squeeze_on.shift(1, fill_value=False) & ~out.squeeze_on & (out.bb_width > out.bb_width.shift(1))
    out["chop"] = choppiness(out, 14)
    out["supertrend"], _ = supertrend(out, 10, 3.0)
    super_bull = out.close > out.supertrend

    out["roc10"] = roc(out.close, 10)
    out["momentum_score"] = pd.concat([
        _sign(out.rsi14 - 50), _sign(out.macd_hist), _sign(out.cci20),
        _sign(out.momentum10), _sign(out.stoch_rsi_k - out.stoch_rsi_d), _sign(out.willr + 50),
    ], axis=1).mean(axis=1)

    # Python runner uses chart-timeframe trend context. The Pine strategy adds a
    # confirmed higher-timeframe vote and previous-day pivot as extra guards.
    out["trend_extra"] = pd.concat([
        _sign(out.ema20 - out.ema50), _sign(out.ema50 - out.ema200),
        _sign(out.close - out.vwap), pd.Series(np.where(super_bull, 1.0, -1.0), index=out.index),
    ], axis=1).mean(axis=1)

    out["direction_score"] = (
        0.28 * out.ma_rating + 0.18 * out.osc_rating + 0.22 * out.trend_extra +
        0.17 * out.momentum_score + 0.15 * out.volume_score
    )

    out["trend_regime"] = (
        (out.adx >= cfg.trend_adx_min) & (out.chop <= cfg.trend_chop_max) &
        (out.direction_score.abs() >= cfg.score_threshold * 0.70)
    )
    out["range_regime"] = (out.adx <= cfg.range_adx_max) & (out.chop >= cfg.range_chop_min)
    out["volatility_ok"] = out.atr_pct <= cfg.max_atr_pct

    don_high = out.high.rolling(cfg.donchian_length).max().shift(1)
    don_low = out.low.rolling(cfg.donchian_length).min().shift(1)
    inside = (out.high < out.high.shift(1)) & (out.low > out.low.shift(1))
    inside_long = inside.shift(1, fill_value=False) & (out.close > out.high.shift(1))
    inside_short = inside.shift(1, fill_value=False) & (out.close < out.low.shift(1))
    outside = (out.high > out.high.shift(1)) & (out.low < out.low.shift(1))
    outside_long = outside & (out.close > out.open) & (out.close > out.close.shift(1))
    outside_short = outside & (out.close < out.open) & (out.close < out.close.shift(1))

    liquid = out.rel_vol >= cfg.min_relative_volume
    super_bear = out.close < out.supertrend

    pull_l = (
        out.trend_regime & (out.direction_score >= cfg.score_threshold) & (out.close > out.vwap) &
        (out.ema20 > out.ema50) & ((out.low <= out.ema20) | (out.low <= out.vwap)) &
        (out.close > out.ema20) & out.rsi14.between(45, 72) & liquid &
        (out.plus_di > out.minus_di) & super_bull
    )
    pull_s = (
        out.trend_regime & (out.direction_score <= -cfg.score_threshold) & (out.close < out.vwap) &
        (out.ema20 < out.ema50) & ((out.high >= out.ema20) | (out.high >= out.vwap)) &
        (out.close < out.ema20) & out.rsi14.between(28, 55) & liquid &
        (out.minus_di > out.plus_di) & super_bear
    )
    br_l = (
        out.volatility_ok & (out.trend_regime | out.squeeze_release) &
        (out.direction_score >= cfg.score_threshold) & (out.rel_vol >= cfg.breakout_relative_volume) &
        (out.bb_width > out.bb_width.shift(1)) & (out.macd_hist > 0) & (out.close > out.vwap) &
        ((out.close > don_high) | inside_long | outside_long)
    )
    br_s = (
        out.volatility_ok & (out.trend_regime | out.squeeze_release) &
        (out.direction_score <= -cfg.score_threshold) & (out.rel_vol >= cfg.breakout_relative_volume) &
        (out.bb_width > out.bb_width.shift(1)) & (out.macd_hist < 0) & (out.close < out.vwap) &
        ((out.close < don_low) | inside_short | outside_short)
    )
    mean_l = (
        out.range_regime & out.volatility_ok & (out.bb_pctb.shift(1) < 0.05) &
        (out.bb_pctb > out.bb_pctb.shift(1)) & (out.rsi14 < 38) & (out.mfi14 < 40) &
        (out.close < out.vwap) & (out.cmf20 > out.cmf20.shift(1))
    )
    mean_s = (
        out.range_regime & out.volatility_ok & (out.bb_pctb.shift(1) > 0.95) &
        (out.bb_pctb < out.bb_pctb.shift(1)) & (out.rsi14 > 62) & (out.mfi14 > 60) &
        (out.close > out.vwap) & (out.cmf20 < out.cmf20.shift(1))
    )

    out["long_setup"] = np.select([br_l, pull_l, mean_l], ["BREAKOUT", "TREND", "MEAN"], default="")
    out["short_setup"] = np.select([br_s, pull_s, mean_s], ["BREAKOUT", "TREND", "MEAN"], default="")
    out["signal"] = np.select(
        [(out.long_setup != "") & (out.short_setup == ""), (out.short_setup != "") & (out.long_setup == "")],
        [1, -1], default=0
    ).astype(int)
    out["setup"] = np.where(out.signal > 0, out.long_setup, np.where(out.signal < 0, out.short_setup, ""))
    return out


def setup_risk(cfg: StrategyConfig, setup: str):
    if setup == "BREAKOUT":
        return cfg.breakout_stop_atr, cfg.breakout_target_r
    if setup == "MEAN":
        return cfg.mean_stop_atr, cfg.mean_target_r
    return cfg.trend_stop_atr, cfg.trend_target_r
