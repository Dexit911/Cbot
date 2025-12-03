import os
import pandas as pd
import json
from visualizer import GraphVision


import os
import pandas as pd
import json
from visualizer import GraphVision


class DataManager:
    def __init__(self, test_data):
        self.result_list = test_data

    def _create_file_name(self):
        """Creates all the names based on the test"""
        sector = self.result_list[0]
        crypto = sector["crypto"]
        strategy = sector["strategy"]
        date = sector["date"]

        main_folder = f"{crypto[:3]}_{strategy}_{date}"
        return main_folder

    def save_results(self):
        """Structure, creates, saves all file"""
        summary_data = self._create_summary_table()
        strategy_data = self._create_strategy_json()

        # MAIN TEST FOLDER
        main_folder = self._create_file_name()
        os.makedirs(main_folder, exist_ok=True)

        # SUMMARY + STRATEGY PATHS
        summary_path = os.path.join(main_folder, "summary.csv")
        strategy_path = os.path.join(main_folder, "strategy.json")

        # SAVE SUMMARY
        summary_data.to_csv(summary_path, index=False)

        # SAVE STRATEGY CONFIG
        with open(strategy_path, "w") as f:
            f.write(strategy_data)

        # LOOP SECTORS
        for i, sector in enumerate(self.result_list):

            # SECTOR FOLDER
            sector_folder_path = os.path.join(main_folder, f"sector-{i}")
            os.makedirs(sector_folder_path, exist_ok=True)

            # PATHS
            sector_data_path = os.path.join(sector_folder_path, "data.csv")
            sector_chart_path = os.path.join(sector_folder_path, "chart.html")

            # Candles-ind + trades
            df_candles_ind = sector["candles-ind"]
            trades = sector["trades"]

            df = self._create_sector_table(i)
            df.to_csv(sector_data_path, index=False)

            print(df_candles_ind.head())

            # Use visualizer class to save chart
            visualizer = GraphVision()
            visualizer.save_chart(df_candles_ind, trades, sector_chart_path)

    def _create_summary_table(self):
        """Creates df for summary csv"""
        df_columns = [
            "profit-factor", "win-rate(%)",
            "biggest-loss(%)", "avg-profit(%)",
            "sectors", "candles-per-sector",
            "start-balance(usdt)", "total-profit(usdt)",
            "total-trades", "trades-per-day",
            "test-length(days)"
        ]

        data = DataCalculator.summary_data(self.result_list)
        df = pd.DataFrame([data], columns=df_columns)
        return df

    def _create_strategy_json(self):
        """Converts strategy config into json"""
        return json.dumps(self.result_list[0]["strategy-config"], indent=2, sort_keys=True)

    def _create_sector_table(self, sector_index: int):
        """Creates a light DF with metadata (drops heavy columns)"""

        sector = self.result_list[sector_index]
        df = pd.DataFrame([sector])

        df.drop(
            columns=[
                "strategy",
                "strategy-config",
                "date",
                "candles-ind",
                "trades"
            ],
            inplace=True,
            errors="ignore"
        )

        return df



class DataCalculator:

    @staticmethod
    def summary_data(result_data: list) -> dict:
        """Calculates summary data"""

        biggest_sector = result_data[0]
        balance_history = biggest_sector["balance-history"]

        profit_factor = biggest_sector["profit-factor"]
        win_rate = biggest_sector["win-rate"]
        biggest_loss = DataCalculator.biggest_loss_percent(balance_history)
        avg_profit = DataCalculator.average_profit_percent(balance_history)

        sectors = len(result_data)
        candles_per_sector = "NaN"
        start_balance = balance_history[0]
        total_profit = biggest_sector["total-profit"]
        total_trades = biggest_sector["total-trades"]
        trades_per_day = "NaN"
        days_length = biggest_sector["days"]

        return {
            "profit-factor": profit_factor,
            "win-rate(%)": win_rate,
            "biggest-loss(%)": biggest_loss,
            "avg-profit(%)": avg_profit,
            "trend": biggest_sector["trend"],
            "sectors": sectors,
            "candles-per-sector": candles_per_sector,
            "start-balance(usdt)": start_balance,
            "total-profit(usdt)": total_profit,
            "total-trades": total_trades,
            "trades-per-day": trades_per_day,
            "test-length(days)": days_length
        }

    @staticmethod
    def biggest_loss_percent(balance_history: list) -> float:
        """Peak loss in percent"""
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
        """Avg profit in percent"""
        balances = balance_history

        if len(balances) < 2:
            return 0

        profits = []
        for i in range(1, len(balances)):
            profit = (balances[i] - balances[i - 1]) / balances[i - 1] * 100
            profits.append(profit)

        return round(sum(profits) / len(profits), 2)

    @staticmethod
    def trades_per_day():
        pass

    @staticmethod
    def define_trend(candles, threshold=0.01):
        """Defines trend: bull / bear / sideways"""
        import numpy as np
        from scipy.stats import linregress

        if isinstance(candles, list):
            prices = [c["close"] for c in candles]
        else:
            prices = candles

        x = np.arange(len(prices))
        slope, _, _, _, _ = linregress(x, prices)

        if slope > threshold:
            label = "bull"
        elif slope < -threshold:
            label = "bear"
            label = "bear"
        else:
            label = "sideways"

        return round(slope, 6), label

