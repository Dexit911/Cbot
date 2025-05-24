import time

from config import *
import pandas as pd
from data_logger import CsvLogger


class TradingLogic:
    """
    Here is going to be all BUY/SELL logic.
    Analyze data, apply indicators and update the DataFrame table.
    Hi Hampus
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol

        self.trades = []

        self.rsi_period = 30
        self.rsi_values = [42, 68]

    def start_simulation(self, balance):

        usdt = balance  # starting money
        crypto = 0  # how much XRP you own
        entry_price = None

        for i in range(RSI_PERIOD, len(self.df)):
            row = self.df.iloc[i]
            current_price = row["Close"]
            current_time = row["Close Time"]
            rsi = row["RSI"]

            if rsi < self.rsi_values[0] and crypto == 0:
                # BUY
                crypto = usdt / current_price
                entry_price = current_price
                usdt = 0

                # DOCUMENT THE BUY TRADE
                self.trades.append({
                    "type": "BUY",
                    "price": current_price,
                    "time": current_time})
                print(f"[{current_time}] BUY at {current_price:.4f}")

            elif rsi > self.rsi_values[1] and crypto > 0:
                # SELL
                usdt = crypto * current_price
                profit = usdt - (entry_price * crypto)
                crypto = 0

                # DOCUMENT THE SELL TRADE
                self.trades.append({
                    "type": "SELL",
                    "price": current_price,
                    "time": current_time,
                    "profit": profit})
                print(f"[{current_time}] SELL at {current_price:.4f} | Profit: {profit:.2f} | Balance: {usdt:.2f} USDT")

        # GET END MONEY RESULT
        usdt = crypto * self.df["Close"].iloc[-1]  # End money
        total_profit = usdt - balance  # The profit of the test
        print(f"balance: {balance}, USDT: {usdt}")  # Debug


        # SAVE DATA
        data = self.get_trade_stats(total_profit)
        CsvLogger.save_test_result("test_trades.csv", data)

        return round(usdt, 2)  # Return the USDT amount

    def get_trade_stats(self, total_profit):
        sell_trades = [t for t in self.trades if t["type"] == "SELL"]  # List with sell trades
        wins = [t for t in sell_trades if t["profit"] > 0]  # How many sell trades are profit
        losses = [t for t in sell_trades if t["profit"] <= 0]  # How many sell trades are loose
        total_trades = len(wins) + len(losses)  # Total trades maked

        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0  # Get win rate in %
        #total_profit = sum(t["profit"] for t in self.trades if t["type"] == "SELL")  # Not working for some reason


        gross_profit = sum(t["profit"] for t in wins)  # How many USDT did you go profit
        gross_loss = abs(sum(t["profit"] for t in losses))  # How many USDT you lost
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")  # Profit factor can be infinity

        self.trades.clear()  # Empty the trade list for the next test

        return {  # return the data
            "Crypto": self.symbol,
            "RSI period (days)": self.rsi_period,
            "RSI values": f"{self.rsi_values[0], self.rsi_values[1]}",
            "Win Rate (%)": round(win_rate, 2),
            "Total profit": round(total_profit, 2),
            "Number of trades": total_trades,
            "Number of candles": len(self.df),
            "Days": round(len(self.df) * 15 / 60 / 24, 2),
            "Profit Factor": round(profit_factor, 1),
            "Date": f"{pd.Timestamp.now()}"[:-7]
        }

    """RSI"""

    def calculate_rsi(self, close_prices):
        """Calculate RSI with given period and close prices"""
        period = self.rsi_period
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_rsi_indicator(self):
        """Apply RSI to the DataFrame"""
        self.df["RSI"] = self.calculate_rsi(self.df["Close"])
