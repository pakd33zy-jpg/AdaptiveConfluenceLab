# Indicator manifest

The Pine strategy directly or functionally incorporates these indicator families:

### Trend / averages
SMA 10/20/30/50/100/200, EMA 10/20/30/50/100/200, HMA 9, VWMA 20, Ichimoku 9/26/52, VWAP, Supertrend, previous-day pivot, confirmed higher-timeframe EMA/RSI.

### Momentum / oscillators
RSI 14, Stochastic, CCI 20, DMI + ADX, Awesome Oscillator, Momentum 10, MACD 12/26/9, Stochastic RSI, Williams %R, Bull Power, Bear Power, Ultimate Oscillator.

### Volume / money flow
Relative Volume, MFI 14, CMF 20, OBV + OBV EMA, VWMA-vs-SMA, VWAP.

### Volatility / regime
ATR 14, ATR%, Bollinger Bands, Bollinger %B, Bollinger BandWidth, Keltner Channel, BB/KC squeeze, Choppiness Index.

### Price structure
20-bar Donchian/Price Channel, inside-bar breakout, outside-bar momentum, previous-day pivot.

The system uses more than thirty components, but they are collapsed into a handful of orthogonal group scores so correlated indicators do not receive unlimited extra weight.
