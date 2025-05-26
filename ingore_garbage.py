class TradingLogic:
    """
    Here is going to be all BUY/SELL logic.
    Analyze data, apply indicators and update the DataFrame table.
    Hi Hampus
    """

    def __init__(self, df, symbol):
        # DATA AND TRADES
        self.df = df
        self.trades = []

        # TEST
        self.test_sectors = 10
        # SIMULATION VALUES
        self.symbol = symbol

        self.start_balance = 0
        self.usdt = 0
        self.crypto = 0
        self.entry_price = 0

        # TRADING LOGIC SETTINGS
        self.rsi_period = 20
        self.rsi_values = [30, 70]

    def _handle_transaction(self, row, action):
        """Handles buy and sell"""
        price = row["Close"]
        time = row["Close Time"]
        if action == "BUY":
            self.crypto = self.usdt / price
            self.entry_price = price
            self.usdt = 0
            self.trades.append({
                "type": "BUY",
                "price": price,
                "time": time})

            # DEBUG
            self.print_debug("BUY", [time, price])

        elif action == "SELL":
            profit = (price - self.entry_price) * self.crypto
            self.usdt = self.crypto * price
            self.crypto = 0
            self.trades.append({
                "type": "SELL",
                "price": price,
                "time": time,
                "profit": profit})

            # DEBUG
            self.print_debug("SELL", [time, price, profit])

            return profit
        return None

    def start_simulation(self, raw_df, start_balance):
        """Handles the start logic"""
        # SET THE START BALANCE
        self.start_balance = start_balance
        # MAKE SUE TO NOT CHANGE ORIGINAL
        df = raw_df.copy()
        # CREATE SECTORS FOR TEST
        test_sectors = self.create_test_sectors(df)
        whole_sector = df
        # RUN TEST ON WHOLE SECTOR (ALL CANDLES)
        self.simulate(whole_sector)
        # RUN TEST ON SECTORS
        for sector in test_sectors:
            self.simulate(sector)

    def simulate(self, sector):
        """Runs the simulation, all trading happens here"""
        print(sector)
        self.df = sector
        self.usdt = self.start_balance  # starting money
        self.crypto = 0  # how much XRP you own
        self.entry_price = None

        # START TRADING
        for i in range(self.rsi_period, len(self.df)):
            row = self.df.iloc[i]
            rsi = row["RSI"]
            if rsi < self.rsi_values[0] and self.crypto == 0:
                self._handle_transaction(row, "BUY")
            elif rsi > self.rsi_values[1] and self.crypto > 0:
                self._handle_transaction(row, "SELL")

        # PERFORM THE END
        self.end_simulation()

    def end_simulation(self):
        """Saves data, performs end logic if needed"""
        last_trade = self.trades[-1] if self.trades else None
        if last_trade and last_trade["type"] == "BUY":
            # IF LAST WAS BUY -> SELL -> GET PROFIT
            self.usdt = self.crypto * self.df["Close"].iloc[-1]
            total_profit = self.usdt - self.start_balance
        else:
            # IF LAST WAS SELL -> GET PROFIT
            total_profit = self.usdt - self.start_balance
        # SAVE DATA
        data = self.get_trade_stats(total_profit)
        CsvLogger.save_test_result("test_trades.csv", data)
        # DEBUG
        self.print_debug("END")
        # CLEAR TRADES FOR NEXT SIMULATION
        self.trades.clear()

    def get_trade_stats(self, total_profit) -> dict:
        sell_trades = [t for t in self.trades if t["type"] == "SELL"]  # List with sell trades
        wins = [t for t in sell_trades if t["profit"] > 0]  # How many sell trades are profit
        losses = [t for t in sell_trades if t["profit"] <= 0]  # How many sell trades are loose
        total_trades = len(wins) + len(losses)  # Total trades maked
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0  # Get win rate in %
        print(f"wins: {len(wins)}, losses: {len(losses)}, total trades: {total_trades}")
        print(f"win_rate: {win_rate}")
        gross_profit = sum(t["profit"] for t in wins)  # How many USDT did you go profit
        gross_loss = abs(sum(t["profit"] for t in losses))  # How many USDT you lost
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")  # Profit factor can be infinity

        return {  # return the data
            "Crypto": self.symbol,
            "RSI period (days)": self.rsi_period,
            "RSI values": f"{self.rsi_values[0], self.rsi_values[1]}",
            "Win Rate (%)": round(win_rate, 2),
            "Total profit": round(total_profit, 2),
            "Number of trades": total_trades,
            "Number of candles": len(self.df),
            "Days": round(len(self.df) * 15 / 60 / 24, 2),
            "Profit Factor": round(profit_factor, 1),
            "Date": f"{pd.Timestamp.now()}"[:-7]
        }

    def print_debug(self, action: str, values: list = None) -> print:
        match action:
            case "END":
                print(f"\nstart balance: {self.start_balance},"
                      f" current USDT: {self.usdt},"
                      f" RSI PERIOD: {self.rsi_period},"
                      f" RSI VALUES: {self.rsi_values}\n")
            case "BUY":

                print(f"[{values[0]}] BUY at {values[1]:.4f}")

            case "SELL":
                print(
                    f"[{values[0]}] SELL at {values[1]:.4f} | Profit: {values[2]:.2f} | Balance: {self.usdt:.2f} USDT")