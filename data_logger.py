import os
import pandas as pd
import json


class DataManager:
    def __init__(self, test_data):
        self.result_list = test_data

    def create_file_name(self):
        """Creates all the names based on the test"""
        sector = self.result_list[0]
        crypto = sector["Crypto"]
        strategy = sector["Strategy"]
        date = sector["Date"]

        main_folder = f"{crypto[:3]}_{strategy}_{date}"
        return main_folder

    def save_results(self):
        """Structure, creates, saves all file"""
        # SET THE DATA
        summary_data = self.create_summary_table()  # DataFrame
        strategy_data = self.create_strategy_json()

        # CREATE MAIN TEST FOLDER
        main_folder = self.create_file_name()
        os.makedirs(main_folder, exist_ok=True)

        # CREATE PATH FOR SUMMARY AND STRATEGY
        summary_path = os.path.join(main_folder, "summary.csv")
        strategy_path = os.path.join(main_folder, "strategy.json")

        # SAVE SUMMARY CSV
        summary_data.to_csv(summary_path, index=False)

        # SAVE STRATEGY SETTINGS INTO A JSON
        with open(strategy_path, "w") as f:
            f.write(strategy_data)

        for i in range(len(self.result_list)):
            # CREATE FOLDER FOR EVERY SECTOR
            sector_folder_path = os.path.join(main_folder, f"Sector_{i}")
            os.makedirs(sector_folder_path, exist_ok=True)

            # CREATE PATHS FOR FULL DATA AND CHART
            sector_data_path = os.path.join(sector_folder_path, "data.csv")
            sector_chart_path = os.path.join(sector_folder_path, "chart.png")

            # SAVE CSV
            df = self.create_sector_table(i)
            df.to_csv(sector_data_path, index=False)

            # SAVE CHART PNG
            with open(sector_chart_path, "w") as f:
                pass

        print("**SAVED DATA** ")

    def create_summary_table(self):
        """Cerates df for summary csv"""
        # SETTING FOR FIXED IN PLACE COLUMNS FOR DATAFRAME
        df_columns = [
            "Profit Factor", "Win Rate(%)",
            "Biggest loss(%)", "Avg profit(%)",
            "Sectors", "Candles per sector",
            "Start Balance(USDT)", "Total Profit(USDT)",
            "Total trades", "Trades per day",
            "Test length(days)",
        ]
        # GET DATA -> TRANSFORM TO DATAFRAME -> RETURN
        data = DataCalculator.summary_data(self.result_list)
        df = pd.DataFrame([data], columns=df_columns)
        return df

    def create_strategy_json(self):
        """Converts strategy config into json"""
        return json.dumps(self.result_list[0]["Strategy config"], indent=2, sort_keys=True)

    def create_sector_table(self, sector_index: int):
        """
        Adjusts sector data and creates a Dataframe
        :param sector_index: decides which sector
        :return: df (pd.DataFrame)
        """
        sector = self.result_list[sector_index]
        df = pd.DataFrame([sector])
        df.drop(columns=["Strategy", "Strategy config", "Date"])  # Remove some columns
        return df


class DataCalculator:

    @staticmethod
    def summary_data(result_data: list) -> dict:
        """
        Calculates all summary data. Used before creating a DataFrame
        :param result_data: list of dicts. On dict has data about one sector
        :return: dict with summary data
        """
        biggest_sector = result_data[0]
        balance_history = biggest_sector["Balance history"]

        # Parse all needed data
        profit_factor = biggest_sector["Profit factor"]
        win_rate = biggest_sector["Win rate(%)"]
        biggest_loss = DataCalculator.biggest_loss_percent(balance_history)
        avg_profit = DataCalculator.average_profit_percent(balance_history)
        sectors = len(result_data)
        candles_per_sector = "NaN"
        start_balance = balance_history[0]
        total_profit = biggest_sector["Total profit"]
        total_trades = biggest_sector["Total trades"]
        trades_per_day = "NaN"
        days_length = biggest_sector["Days"]

        return {
            "Profit Factor": profit_factor,
            "Win Rate(%)": win_rate,
            "Biggest loss(%)": biggest_loss,
            "Avg profit(%)": avg_profit,
            "Trend": biggest_sector["Trend"],
            "Sectors": sectors,
            "Candles per sector": candles_per_sector,
            "Start Balance(USDT)": start_balance,
            "Total Profit(USDT)": total_profit,
            "Total trades": total_trades,
            "Trades per day": trades_per_day,
            "Test length(days)": days_length,

        }

    @staticmethod
    def biggest_loss_percent(balance_history: list) -> float:
        """
        Calculate the peak loss in percent
        :param balance_history: list of floats.
        :return biggest loss in percent (float)
        """
        balances = balance_history
        peak = balances[0]
        max_drawdown = 0
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (balance - peak) / peak * 100
            max_drawdown = min(max_drawdown, drawdown)
        return round(max_drawdown, 2)

    @staticmethod
    def average_profit_percent(balance_history: list) -> float:
        """
        Calculates average profit percent
        :param balance_history: list of floats
        :return: average profit percent (float)
        """
        balances = balance_history
        if len(balances) < 2:
            return 0
        profits = []
        for i in range(1, len(balances)):
            # Calculate % change between each balance point
            profit = (balances[i] - balances[i - 1]) / balances[i - 1] * 100
            profits.append(profit)
        return round(sum(profits) / len(profits), 2)

    @staticmethod
    def trades_per_day():
        pass

    @staticmethod
    def define_trend(candles, threshold=0.01):
        """
        :param candles: list of dicts with "Close" prices (or a DataFrame column)
        :param threshold: slope limit to classify as sideways
        :return: slope (float), label (str)
        """
        import numpy as np
        from scipy.stats import linregress

        if isinstance(candles, list):
            prices = [c["Close"] for c in candles]
        else:
            prices = candles  # e.g. a pd.Series

        x = np.arange(len(prices))
        slope, _, _, _, _ = linregress(x, prices)

        if slope > threshold:
            label = "bull"
        elif slope < -threshold:
            label = "bear"
        else:
            label = "sideways"

        return round(slope, 6), label
