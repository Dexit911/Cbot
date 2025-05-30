import time
from data_fetcher import BinanceFetcher as Bf
from indicators import TestRunner, BaseStrategy, RSIStrategy
from config import CRYPTO_LIST

"""SIMULATION ONLY TAKES CSV FORMAT DATA!!!!!"""


def test():
    symbol = "XRPUSDT"
    df = Bf(symbol=symbol).get_data(20000)
    csv_format = Bf.csv_reformat(df)

    # CREATE AND SET TESTRUNNER
    test_field = TestRunner(
        df=csv_format,
        strategy_id="RSIStrategy",
        symbol=symbol,
        start_balance=1000,
        test_sectors=10
    )

    # START THE SIMULATION
    test_field.start_simulation()


test()
