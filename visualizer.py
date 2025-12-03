import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GraphVision:
    def __init__(self, signal_size=10):
        self.signal_size = signal_size

    # === FORMAT DATAFRAME ===
    @staticmethod
    def _format_df(df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df = df.set_index("Open Time")
        df.index.name = "Date"
        return df

    # === DETECT INDICATORS ===
    @staticmethod
    def _detect_indicators(df: pd.DataFrame) -> dict:
        return {
            "has_rsi": "RSI" in df.columns,
            "has_ma": "MA" in df.columns,
            "has_bb": all(col in df.columns for col in ["BB_Mid", "BB_Upper", "BB_Lower"])
        }

    # === PLOT PRICE + MA + BB ===
    def _plot_price(self, fig, df, indicators):
        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles"
            ),
            row=1, col=1
        )

        # MA
        if indicators["has_ma"]:
            ma = df.dropna(subset=["MA"])
            fig.add_trace(
                go.Scatter(
                    x=ma.index,
                    y=ma["MA"],
                    mode="lines",
                    line=dict(width=2, color="orange"),
                    name="MA"
                ),
                row=1, col=1
            )

        # Bollinger Bands
        if indicators["has_bb"]:
            bb = df.dropna(subset=["BB_Upper", "BB_Lower", "BB_Mid"])

            # Mid line
            fig.add_trace(
                go.Scatter(
                    x=bb.index,
                    y=bb["BB_Mid"],
                    mode="lines",
                    line=dict(width=2, color="cyan", dash="dot"),
                    opacity=0.7,
                    name="BB Mid"
                ),
                row=1, col=1
            )

            # Upper
            fig.add_trace(
                go.Scatter(
                    x=bb.index,
                    y=bb["BB_Upper"],
                    mode="lines",
                    line=dict(width=2, color="lightgray"),
                    opacity=0.3,
                    name="BB Upper"
                ),
                row=1, col=1
            )

            # Lower
            fig.add_trace(
                go.Scatter(
                    x=bb.index,
                    y=bb["BB_Lower"],
                    mode="lines",
                    line=dict(width=2, color="lightgray"),
                    opacity=0.3,
                    name="BB Lower"
                ),
                row=1, col=1
            )

            # Shading between bands
            x = pd.concat([pd.Series(bb.index), pd.Series(bb.index[::-1])])
            y = pd.concat([bb["BB_Upper"], bb["BB_Lower"][::-1]])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    fillcolor="rgba(200,200,200,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=1
            )

    # === PLOT RSI ===
    def _plot_rsi(self, fig, df, indicators):
        if not indicators["has_rsi"]:
            return

        rsi = df.dropna(subset=["RSI"])

        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi["RSI"],
                mode="lines",
                line=dict(width=1, color="yellow"),
                name="RSI"
            ),
            row=2, col=1
        )

    # === PLOT BUY/SELL SIGNALS ===
    def _plot_signals(self, fig, trades):
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for t in trades:
            try:
                t_time = pd.to_datetime(t["time"], unit="ms")
            except:
                continue

            side = t["type"].lower()

            if side == "buy":
                buy_x.append(t_time)
                buy_y.append(t["price"])

            elif side == "sell":
                sell_x.append(t_time)
                sell_y.append(t["price"])

        # BUY markers
        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode="markers+text",
                    marker=dict(
                        color="lime",
                        size=self.signal_size,
                        symbol="triangle-up",
                        line=dict(width=3, color="black"),
                    ),
                    name="Buy"
                ),
                row=1, col=1
            )

        # SELL markers
        if sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode="markers+text",
                    marker=dict(
                        color="red",
                        size=self.signal_size,
                        symbol="triangle-down",
                        line=dict(width=3, color="black"),
                    ),
                    name="Sell"
                ),
                row=1, col=1
            )

    # === MAIN METHOD ===
    def save_chart(self, df_binance: pd.DataFrame, trades: list, file_path: str) -> None:
        df = self._format_df(df_binance)
        indicators = self._detect_indicators(df)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25]
        )

        # Plot price indicators (candles, MA, BB)
        self._plot_price(fig, df, indicators)

        # Plot RSI subplot
        self._plot_rsi(fig, df, indicators)

        # Plot Buy/Sell signals
        self._plot_signals(fig, trades)

        # Layout styling
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=650,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )

        # Export interactively
        fig.write_html(file_path)
