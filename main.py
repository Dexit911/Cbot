import time

from data_logger import CsvLogger
from data_fetcher import BinanceFetcher
from indicators import TradingLogic
from config import RSI_PERIOD, LIMIT, CRYPTO_LIST


def main():
    # TYPE IN VALUES
    balance = float(input("Your USDT bet: "))
    candles = int(input("How many candles back in simulation you want to go? (One candle is 15 min): "))

    # CONVERT TIME
    hours = candles * 15 / 60
    days = hours / 24

    # DEFINE SAVE AND DEFINE DATAFRAMES, DEFINE LOGIC
    for symbol in CRYPTO_LIST:
        time.sleep(0.2)
        df = BinanceFetcher(symbol=symbol).get_data(candles)  # Get data and create table
        logic = TradingLogic(df, symbol)  # Get the trade logic
        logic.apply_rsi_indicator()  # Apply RSI to the table
        #CsvLogger.save(df, "xrp_data.csv")  # Save in Csv

        # START SIMULATION
        result = logic.start_simulation(balance)  # simulate and get answer

        # PRINT THE RESULTS
        if result:
            print(
                f"You started with {balance} USDT and ended up with {result} USDT\n\nIt took {hours} hours or {days} days")
        else:
            print("You lost all of you money")


main()
