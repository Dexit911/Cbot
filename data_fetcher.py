import pandas
import requests
import pandas as pd
from config import *
import time


class BinanceFetcher:
    def __init__(self, symbol=SYMBOL, interval=INTERVAL):
        self.url = f"https://api.binance.com/api/v3/klines"
        self.interval = interval  # The candle interval (15m)
        self.symbol = symbol  # Which crypto to trade
        self.limit = 1000  # Max candles BINANCE lets you fetch

    def fetch_data(self, total_candles) -> list:
        tc = total_candles  # How many candles you want to fetch
        url = self.url  # Binance url
        limit = self.limit  # Binance limit
        interval = self.interval
        all_data = []

        now = int(time.time() * limit)
        step_ms = self.interval_to_milliseconds(limit, interval) * limit
        end_time = now

        while len(all_data) < tc:
            time.sleep(0.2)
            start_time = end_time - step_ms  # Debug

            params = {  # The params for the request
                "symbol": self.symbol,  # Crypto ("XRPUSDT")
                "interval": self.interval,  # Interval for one candle ("15m"
                "limit": limit,  # Limit for candles is allways max (1000)
                "endTime": end_time  # timestamp in ms to end the data at
            }

            response = requests.get(url, params=params).json()
            if not response:
                break
            all_data = response + all_data
            end_time = response[0][0]
        return all_data

    def get_data(self, total_candles) -> pandas.DataFrame:
        json = self.fetch_data(total_candles)  # Fetch the data and get the json
        df = pd.DataFrame(json, columns=[  # Append the json to the DataFrame
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Volume", "Trades",
            "Taker Buy Base", "Taker Buy Quote", "Ignore"
        ])
        self.reformat_data(df)  # Make the table cleaner
        return df

    # STATIC
    @staticmethod
    def reformat_data(df):
        """"Gives cleaner format to data"""
        remove_zero_list = [
            "Open", "High", "Low", "Close", "Volume", "Quote Volume", "Taker Buy Quote", "Taker Buy Base"
        ]
        reformat_time_list = ["Open Time", "Close Time"]
        # Remove the zeros from the long float
        for column in remove_zero_list:
            df[column] = pd.to_numeric(df[column])
        # Convert to real time format
        for column in reformat_time_list:
            df[column] = pd.to_datetime(df[column], unit="ms")

    @staticmethod
    def interval_to_milliseconds(limit, interval):
        unit = interval[-1]
        amount = int(interval[:-1])
        if unit == 'm':
            return amount * 60 * limit
        elif unit == 'h':
            return amount * 60 * 60 * limit
        elif unit == 'd':
            return amount * 24 * 60 * 60 * limit
        raise ValueError("Unsupported interval format")
