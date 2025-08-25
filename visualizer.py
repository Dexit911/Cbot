import mplfinance as mpf
import numpy as np
import pandas as pd
import os


class GraphVision:

    @staticmethod
    def mpf_reformat(df_candles: pd.DataFrame) -> pd.DataFrame:

        """
        Makes DataFrame ready for mpf visualizing
        :param df_candles: pd.DataFrame in csv format, df with candle data
        :return: pd.DataFrame for making chart visualizing
        """
        df = df_candles.copy()  # Make a copy
        df = df.set_index("Open Time")  # Set the Datetime as index for plotting
        df.index.name = "Date"  # Change the name
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)  # Save DataFrame with values needed for mpf
        return df

    @staticmethod
    def save_chart(sector: dict, file_path: str):
        """
        Creates graph png and saves it
        :param sector: dict with all sector result data
        :param file_path: save file in given path (str)
        """

        df_candles = sector["Candles"].copy()  # Make a copy to be safe
        df = GraphVision.mpf_reformat(df_candles)  # Reformat
        add_plots = GraphVision.get_plots(sector)  # Add extra indicators to graph

        candles = len(df)
        graph_type = "candle"
        if candles >= 500:
            graph_type = "line"

        # SAVE PNG
        mpf.plot(
            df, addplot=add_plots, type=graph_type, volume=True, figscale=2,
            savefig=dict(
                fname=os.path.join(file_path, "chart.png"),
                bbox_inches="tight",  # trim extra white space
                dpi=500,  # res
                pad_inches=0.1),  # padding

        )

        # CREATE PATHS
        csv_candles_path = os.path.join(file_path, "temporary_candle_data.csv")
        inspect_script_path = os.path.join(file_path, "inspect.py")
        # SAVE TO TEMPORARY CSV
        df_candles.to_csv(csv_candles_path)
        # CREATE INSPECT SCRIPT
        GraphVision.create_inspect_script(
            add_plots,
            graph_type,
            csv_candles_path,
            inspect_script_path
        )

    @staticmethod
    def create_inspect_script(add_plots, graph_type, from_path, to_path):
        # CREATE SCRIPT STRING
        mpf_script = f"""import mplfinance as mpf
        import pandas as pd
        df = pd.read_csv({from_path})
        mpf.plot(
            df,  
            addplot={add_plots}
            type={graph_type}
            volume=True"""
        # SAVE THE SCRIPT
        with open(to_path, "w") as f:
            f.write(mpf_script)

        os.remove(from_path)

    @staticmethod
    def get_plots(sector: dict) -> list:
        """
        Add other markers to the graph if needed
        :param sector: dict  with all result data
        :return: list with all extra plots to add to graph
        """
        df_candles = sector["Candles"]
        add_plots = []

        if "RSI" in df_candles.columns:
            add_plots.append(GraphVision.get_rsi_plot(df_candles))

        if "BB_Mid" in df_candles.columns:
            add_plots.extend(GraphVision.get_bollinger_plot(df_candles))

        add_plots.extend(GraphVision.get_trade_markers(sector["Trades"], df_candles))

        return add_plots

    @staticmethod
    def get_bollinger_plot(df: pd.DataFrame) -> list:
        """
        Add all three bollinger bands lines to graph
        :param df: pd.DataFrame, candles data
        :return: list with plotting settings
        """
        return [
            mpf.make_addplot(df["BB_Upper"], color="green"),
            mpf.make_addplot(df["BB_Lower"], color="red"),
            mpf.make_addplot(df["BB_Mid"], color="gray")
        ]

    @staticmethod
    def get_rsi_plot(df: pd.DataFrame):
        """
        :param df: pd.DataFrame, candles data
        :return: plotting settings
        """
        return mpf.make_addplot(df["RSI"], panel=1, color="purple", ylabel="RSI")

    @staticmethod
    def get_trade_markers(trades: list, candles_df: pd.DataFrame) -> list:
        """
        Add buy and sell signals based on trades
        :param trades: list with trade dicts
        :param candles_df: pd.DataFrame mpf formatted candle data
        """
        # Generate NaN pd series with length of dates
        buy_signals = pd.Series(np.nan, index=candles_df.index)
        sell_signals = pd.Series(np.nan, index=candles_df.index)

        for trade in trades:
            if trade["type"] == "BUY":
                timestamp = candles_df.index[trade["index"]]
                buy_signals.loc[timestamp] = trade["price"]
            if trade["type"] == "SELL":
                timestamp = candles_df.index[trade["index"]]
                sell_signals.loc[timestamp] = trade["price"]

        print(buy_signals)

        return [
            mpf.make_addplot(buy_signals, type="scatter", marker="^", color="green", markersize=20),
            mpf.make_addplot(sell_signals, type="scatter", marker="v", color="red", markersize=20)
        ]
