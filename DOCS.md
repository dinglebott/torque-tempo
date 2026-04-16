## RAW DATA COLLECTION
Instrument: EUR_USD at 4-hour and 1-hour timeframe (H4, H1)\
Candle data: OHLCV (open, high, low, close, volume)\
Time period: 2005-01-01 to 2026-04-01\
Pulled from OANDA REST-v20 API, stored in JSON format\
Target variable: direction of price movement after *n* candles: Up/down if movement exceeds *x* pips, else flat\
For H4: *n* = 2, *x* = 10\
For H1: *n* = 2, *x* = 4\
<br/>

## DATASETS
Data range: 2005-01-01 to 2026-04-01\
Train/Val/Test split: 0.8/0.1/0.1\
<br/>

## FEATURE ENGINEERING
**Price:**\
Open return => ln(O / previous C)\
High return => ln(H / previous C)\
Low return => ln(L / previous C)\
Close return => ln(C / previous C)\
Volume return => ln(V / previous V)\
**Structure:**\
High-low spread => ln(H - L)\
Open-close spread => ln(C - O)\
Upper wick => (High - candle top) / atr_14\
Lower wick => (Candle bottom - low) / atr_14\
**Trend:**\
15-period EMA => ln(C / EMA)\
50-period EMA\
100-period EMA
EMA cross => ln(ema12 / ema26)\
ADX\
DI diff => plusDI - minusDI\
**UltimateSmoother EMA:**\
14-period UltimateSmoother => ln(C / Smoothed)\
35-period UltimateSmoother\
Smooth cross => ln(smooth8 / smooth18)\
*Applied John Ehler's UltimateSmoother on the close series*\
**Momentum:**\
14-period RSI (smoothed C)\
12/26/9-period MACD histogram => ((ema12 - ema26) - signal) / C\
**Volatility:**\
14-period ATR\
Volatility regime => atr_14 / atr_14 50-period mean\
Bollinger band width => (upperband - lowerband) / midband\
**Volume:**\
Volume ratio => volume / volume sma30\
Volume momentum => vol_ratio - vol_ratio 5-period mean\
**Mean reversion:**\
Bollinger band position => (C - lowerband) / (upperband - lowerband)\
<br/>

## MODEL EVALUATION
**Explanation of metrics:**\
Negative = 0, Flat = 1, Positive = 2\
F1 score (0-1) => Harmonic mean of Precision and Recall\
F1 score (macro-averaged) => Unweighted mean of F1 score calculated for each class (1 and 0)\
Loss score => Cross-entropy loss\
ROC-AUC score (0-1) => Probability that a randomly chosen 1 is ranked higher than a randomly chosen 0 by the model\
Precision (0-1) => Correctly predicted 1's / All predicted 1's\
Recall (0-1) => Correctly predicted 1's / All real 1's\
<br/>