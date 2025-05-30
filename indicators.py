import time
import pandas
from config import *
import pandas as pd
from data_logger import DataManager
import strategies_config as sc

from data_logger import DataCalculator as Dc

"""
STRATEGY SETTINGS

config = {
  "RSIStrategy": {
        "indicators": [
            {"name": "RSI", "params": {"period": 14}},
            {"name": "BollingerBands", "params": {"window": 20, "std": 2}}
        ],
        "params": {
            "rsi_buy": 30,
            "rsi_sell": 70
        }
    },
"""


class IndicatorManager:
    def __init__(self, csv_df):
        """
        :param csv_df: The raw DataFrame in csv format!
        """
        self.df = csv_df

    def apply(self, indicators: list) -> None:
        # ADD INDICATOR BASED ON THE NAME AND ITS SETTINGS
        for ind in indicators:
            name = ind["name"]  # str
            params = ind.get("params", {})
            match name:
                case "RSI":
                    self.apply_rsi(**params)
                case "MACD":
                    self.apply_macd(**params)

    def apply_rsi(self, **kwargs) -> None:
        """Calculates and applies the rsi into DataFrame"""
        period = kwargs.get("period", 14)
        # CALCULATE
        delta = self.df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        self.df["RSI"] = 100 - (100 / (1 + rs))

    def apply_macd(self, **kwargs) -> None:
        pass

    def apply_bollinger(self, **kwargs) -> None:
        pass


class BaseStrategy:
    def __init__(self, config):
        self.config = config

    def apply_indicators(self, df: pandas.DataFrame) -> None:
        """Applies indicators to the raw DataFrame"""
        indicators = self.config["indicators"]
        IndicatorManager(df).apply(indicators)

    def generate_signal(self, row, has_position) -> str:
        """Here does logic comes in. Let the child class decide"""
        pass

    @staticmethod
    def choose_strategy(strategy_id: str):
        """Choose strategy based on id"""
        strategies = {
            "RSIStrategy": RSIStrategy,
        }
        return strategies.get(strategy_id)


class RSIStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        """
        PARAMS FOR STRATEGY
        rsi_buy: the lower index for buy
        rsi_sell: the higher index for sell
        """

    def generate_signal(self, row, has_position) -> str:
        """Small logic for buy and sell with RSI index"""
        # GET THE LOGIC SETTINGS
        params = self.config["params"]
        buy_index = params["rsi_buy"]
        sell_index = params["rsi_sell"]
        # SETTING OUT CONDITIONS
        rsi = row["RSI"]
        if rsi < buy_index and not has_position:
            return "BUY"
        elif rsi > sell_index and has_position:
            return "SELL"
        return "HOLD"


class TestRunner:
    def __init__(
            self,
            df: pandas.DataFrame,
            strategy_id: str,
            symbol: str,
            start_balance: float = 1000.0,
            test_sectors: int = 10
    ):
        """
        :param df: candle data in csv format! Without RSI indicator
        :param strategy_id: what trading logic do you use? Config has all the names and settings.
        :param symbol: which crypto are you trading? Ex "XRPUSDT"
        :param start_balance: how much money do you want to simulate (USDT)?
        :param test_sectors: in how many sectors does the big test is going be sliced up. Avoid using complicated number
        """
        # SET THE START VALUES
        self.main_df = df
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.start_balance = start_balance
        self.test_sectors = test_sectors
        self.sectors = []  # dict -> df, start_index, end_index, label. These are going in the simulation as test samples

        # VALUES USED IN SIMULATION
        self.usdt = 0
        self.crypto = 0
        self.entry_price = 0
        self.has_position = False
        self.trades = []
        self.strategy = None
        self.balances = []

        self.results = []  # List with dicts about every sector + whole sector

    def start_simulation(self):
        """Prepares attributes for simulation"""
        df = self.main_df.copy()  # Make a copy for safety
        config = sc.STRATEGIES[self.strategy_id]  # Get the config by the id
        self.strategy = BaseStrategy.choose_strategy(self.strategy_id)(config)  # Pick the strategy
        self.strategy.apply_indicators(df)  # Apply needed indicators

        # CLEAN TRADE LIST
        self.trades.clear()

        # SECTOR GENERATION
        self.sectors = self.create_sectors(df, 14)
        for sector in self.sectors:
            self.simulate(sector)

        DataManager(self.results).save_results()  # Save results
        self.results.clear()  # Clear results

    def simulate(self, sector: dict) -> None:
        """
        :param sector: dict that stores sector data
        Here is the main simulate loop, that tests all the sectors
        Strategy checks every row. If logic gives a signal - transaction is made.
        """

        sector_dict = sector
        sector_df = sector_dict["df"]

        # RESET THE VALUES
        self.usdt = self.start_balance  # starting money
        self.crypto = 0  # how much XRP you own
        self.entry_price = None
        self.trades = []
        self.has_position = False
        self.balances = [self.start_balance]

        for i in range(len(sector_df)):
            row = sector_df.iloc[i]
            signal = self.strategy.generate_signal(row, self.has_position)
            self._make_transaction(row, signal)

        self.end_simulation(sector_dict)

    def end_simulation(self, sector_dict: dict) -> None:
        """
        Saves data, performs end logic if needed, get things ready to next test
        :param sector_dict:
        """
        sector_df = sector_dict["df"]

        if self.crypto > 0:
            last_price = sector_df["Close"].iloc[-1]
            self._make_transaction({"Close": last_price, "Close Time": "Auto Sell"}, "SELL")

        # PACK ALL RESULTS DATA IN LIST -> SEND TO DATAMANAGER
        data = self.calculate_data(sector_dict)
        self.results.append(data)

    def _make_transaction(self, row, signal):
        """
        Handles Buy and Sell in simulation
        :param row: dict with candle data
        :param signal: str that can be "BUY", "SELL".
        """
        price = row["Close"]
        time = row["Close Time"]
        if signal == "BUY":
            self.has_position = True
            self.crypto = self.usdt / price
            self.entry_price = price
            self.usdt = 0
            self.trades.append({
                "type": "BUY",
                "price": price,
                "time": time})

        elif signal == "SELL":
            self.has_position = False
            profit = (price - self.entry_price) * self.crypto
            self.usdt = self.crypto * price
            self.balances.append(self.usdt)
            self.crypto = 0
            self.trades.append({
                "type": "SELL",
                "price": price,
                "time": time,
                "profit": profit})

    def create_sectors(self, ind_df, lookback=0) -> list:
        """Creates and return list with test sectors
        :param ind_df: base df with indicators
        :param lookback: some indicators needs a lil offset to work, ex RSI has a 14 period.
        """
        sectors = []  # Returning these mfs, wrap them up in a list and throw in simulation
        total_candles = len(ind_df)
        sector_len = int(total_candles / self.test_sectors)

        for sector in range(self.test_sectors):
            logic_start = sector * sector_len
            start = max(0, logic_start - lookback)
            end = logic_start + sector_len
            if end > total_candles:
                break
            # CREATE SECTOR DATA AND APPEND
            sector_df = ind_df.iloc[start:end].copy()  # Just copy to be sure not to fuck up the base DataFrame
            sectors.append({
                "df": sector_df,
                "start_index": start,
                "end_index": end,
                "label": None,  # <----- DANIEL ADD LATER, DO NOT FORGET TS
            })
        # ADD THE BIG FIRST SECTOR TO LIST
        sectors.append({
            "df": ind_df,
            "start_index": total_candles,
            "end_index": 0,
            "label": None,
        })
        return sectors[::-1]  # Flip for better visual

    def calculate_data(self, sector_dict) -> dict:
        """
        Calculates and returns data about a sector
        :param sector_dict: dict with sector data
        :return: dict with all test results about a sector
        """
        df = sector_dict["df"]
        crypto = self.symbol
        sell_trades = [t for t in self.trades if t["type"] == "SELL"]  # List with sell trades
        wins = [t for t in sell_trades if t["profit"] > 0]  # How many sell trades are profit
        losses = [t for t in sell_trades if t["profit"] <= 0]  # How many sell trades are loose
        total_trades = len(wins) + len(losses)  # Total trades maked
        win_rate = round((len(wins) / total_trades) * 100 if total_trades > 0 else 0, 2)  # Get win rate in %
        gross_profit = sum(t["profit"] for t in wins)  # How many USDT did you go profit
        gross_loss = abs(sum(t["profit"] for t in losses))  # How many USDT you lost
        profit_factor = round(gross_profit / gross_loss if gross_loss > 0 else float("inf"), 2)
        total_profit = round(self.usdt - self.start_balance, 2)  # How much did you earn/lose on the whole test
        candle_array = sector_dict["start_index"], sector_dict["end_index"]
        days = round(len(df) * 15 / 60 / 24, 2)  # How days the simulation go back
        date = f"{pd.Timestamp.now()}"[:-16]  # Current time of the test
        strategy_config = sc.STRATEGIES[self.strategy_id]
        trend = Dc.define_trend(df["Close"])  # Define the of current sector

        # DEBUG
        print(f"\n\n----------CANDLE ARRAY: {candle_array}----------\n"
              f"Start balance: {self.start_balance}\n"
              f"Total profit: {total_profit}\n"
              f"Win rate: {win_rate}\n"
              f"Profit factor: {profit_factor}\n"
              f"------------------------------------------")

        return {
            "Crypto": crypto,  # str
            "Strategy": self.strategy_id,  # str
            "Strategy config": strategy_config,  # dict
            "Win rate(%)": win_rate,  # float
            "Total profit": total_profit,  # float
            "Balance history": self.balances,  # list
            "Candle array": candle_array,  # tuple
            "Total trades": total_trades,  # int
            "Profit factor": profit_factor,  # float
            "Days": days,  # float
            "Date": date,  # str
            "Trend": trend
        }
