import pandas
import requests
import pandas as pd
from config import *
import time


class BinanceFetcher:
    def __init__(self, symbol="BTCUSDT", interval="15m"):
        self.url = f"https://api.binance.com/api/v3/klines"
        self.interval = interval  # The candle interval (15m)
        self.symbol = symbol  # Which crypto to trade
        self.limit = LIMIT

    def fetch_data(self, total_candles) -> list:
        all_data = []  # where the fetched data is stored

        tc = total_candles  # How many candles you want to fetch
        url = self.url  # Binance url
        limit = min(self.limit, total_candles - len(all_data))  # Use limit
        interval = self.interval  # interval for candles

        now = int(time.time() * self.limit)
        step_ms = self.interval_to_milliseconds(limit, interval) * limit
        end_time = now

        while len(all_data) < tc:
            time.sleep(0.2)

            params = {  # The params for the request
                "symbol": self.symbol,  # Crypto ("XRPUSDT")
                "interval": self.interval,  # Interval for one candle ("15m")
                "limit": limit,  # Limit for candles is allways max (1000)
                "endTime": end_time  # timestamp in ms to end the data at, always to now
            }

            response = requests.get(url, params=params).json()
            if not response:
                break
            all_data = response + all_data  # sew the data together
            end_time = response[0][0]
        return all_data[-total_candles:]

    def get_data(self, total_candles) -> pandas.DataFrame:
        """Returns DataFrame in exactly same format as Binance"""
        json = self.fetch_data(total_candles)  # Fetch the data and get the json
        df = pd.DataFrame(json, columns=[  # Append the json to the DataFrame
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Volume", "Trades",
            "Taker Buy Base", "Taker Buy Quote", "Ignore"
        ])
        return df

    # STATIC
    @staticmethod
    def csv_reformat(raw_df) -> pandas.DataFrame:
        """"Makes DataFrame ready for csv"""
        df = raw_df.copy()
        numeric_list = ["Open", "High", "Low", "Close", "Volume", "Quote Volume", "Taker Buy Quote", "Taker Buy Base"]
        time_list = ["Open Time", "Close Time"]
        # ADJUST FORMATS
        for column in numeric_list:
            df[column] = pd.to_numeric(df[column])  # Takes out extra zeros
        for column in time_list:
            df[column] = pd.to_datetime(df[column], unit="ms")  # from ms to date time
        return df

    @staticmethod
    def mpf_reformat(raw_df) -> pandas.DataFrame:
        """Makes DataFrame ready for mpf"""
        df = raw_df.copy()  # Make a copy
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")  # Convert Open time from ms -> date time
        df = df.set_index("Open Time")  # Set the Datetime as index for plotting
        df.index.name = "Date"  # Change the name
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)  # Save DataFrame with values needed for mpf
        return df

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
