import os
import pandas as pd


class CsvLogger:
    @staticmethod
    def save(df, filename):
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode="a", index=False, header=not file_exists)

    @staticmethod
    def save_test_result(filename: str, row: dict):
        columns = [
            "Crypto",
            "RSI period (days)",
            "RSI values",
            "Win Rate (%)",
            "Total profit",
            "Number of trades",
            "Number of candles",
            "Days",
            "Profit Factor",
            "Date",
        ]
        df = pd.DataFrame([row])[columns]
        CsvLogger.save(df, filename)
