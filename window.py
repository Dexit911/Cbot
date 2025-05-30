import mplfinance as mpf


class CandleChart:
    def __init__(self, df, title):
        self.df = df
        self.title = title
        self.candles = len(df)

    def save(self):
        df = self.df
        candles = self.candles

        if candles <= 500:
            chart = "candle"
        else:
            chart = "line"

        print(df)

        mpf.plot(self.df, type=chart, volume=True, style="charles", title=self.title)

