import mplfinance as mpf
import pandas as pd


class GraphVision:

    @staticmethod
    def mpf_reformat(df_binance) -> pd.DataFrame:
        print(df_binance)
        """
        Makes DataFrame ready for mpf visualizing
        :param df_binance: pd.DataFrame in the same format as Binance is sending
        :return: pd.DataFrame for making chart visualizing
        """
        df = df_binance.copy()  # Make a copy
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")  # Convert Open time from ms -> date time
        df = df.set_index("Open Time")  # Set the Datetime as index for plotting
        df.index.name = "Date"  # Change the name
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)  # Save DataFrame with values needed for mpf
        return df

    @staticmethod
    def save_chart(df_binance, file_path):
        """
        Creates graph png and saves it
        :param df_binance: pd.DataFrame in the same format as Binance is sending
        :param file_path: save file in given path (str)
        """

        df = GraphVision.mpf_reformat(df_binance)  # Reformat
        candles = len(df)
        type = "candle"
        if candles >= 500: type = "line"
        mpf.plot(  # Save png
            df,
            type=type,
            savefig=dict(fname=file_path, dpi=150, pad_inches=0.2)
        )
