This is a Python-based bot for simulating crypto trades using historical data.
Used to develop and test strategies.

It calculates how much you would gain or lose by trading different cryptocurrencies.

Features
Fetches historical data from Binance
Uses RSI-based strategy (adjustable in TradingLogic.py)
Simulates trades and logs performance

Notes
Do not request more than 20,000 candles at once. This can lead to an IP ban from Binance.
For development and testing purposes only

Files
main.py – Start file
TradingLogic.py – Strategy logic and parameters
test_trades.csv - storing data

Lib 
Pandas
Requests


