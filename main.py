import time
from data_logger import CsvLogger
from data_fetcher import BinanceFetcher as Bf
from indicators import TradingLogic
from config import CRYPTO_LIST
from window import CandleChart

"""SIMULATION ONLY TAKES CSV FORMAT DATA!!!!!"""


def main():
    # TYPE IN VALUES
    balance = float(input("Your USDT bet: "))
    candles = int(input("How many candles back in simulation you want to go? (One candle is 15 min): "))
    print("\n")

    # CONVERT TIME
    hours = candles * 15 / 60
    days = round(hours / 24, 1)

    # DEFINE SAVE AND DEFINE DATAFRAMES, DEFINE LOGIC
    for symbol in CRYPTO_LIST:
        time.sleep(0.2)
        # SETUP THE DATA ADN TRADING LOGIC FOR SIMULATION
        df = Bf(symbol=symbol).get_data(candles)  # Get data and create table
        csv_df = Bf.csv_reformat(df)
        mpf_df = Bf.mpf_reformat(df)

        logic = TradingLogic(csv_df, symbol)  # Get the trade logic
        logic.apply_rsi_indicator()  # Apply RSI to the table
        # CsvLogger.save(df, "xrp_data.csv")  # Save in Csv

        # START SIMULATION
        result = logic.start_simulation(balance)  # simulate and get answer

        # PRINT THE RESULTS
        print(f"You started with {balance} USDT and ended up with {result} USDT\nIt took {hours} hours or {days} days")
        print("\n-----------------------------------------------------------------------------------------------------")

        ch = CandleChart(mpf_df, symbol)

        # ch.start()


def test():
    symbol = "XRPUSDT"
    df = Bf(symbol=symbol).get_data(20000)
    csv_format = Bf.csv_reformat(df)
    logic = TradingLogic(csv_format, symbol)
    logic.apply_rsi_indicator()
    logic.start_simulation(csv_format, 1000)
    mpf_format = Bf.mpf_reformat(df)
    chart = CandleChart(mpf_format, "XRP")
    chart.start()


test()
