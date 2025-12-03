STRATEGIES = {
    "RSIStrategy": {
        "indicators": [
            {"name": "RSI", "params": {"period": 15}},
        ],
        "params": {
            "rsi_buy": 30,
            "rsi_sell": 70
        }
    },

    "BollingerBands": {
        "indicators": [
            {"name": "BB", "params": {"period": 20, "std": 2}}
        ],
        "params": {
        }
    },

    "RSIBB": {
        "indicators": [
            {"name": "RSI", "params": {"period": 14}},
            {"name": "BB", "params": {"period": 20, "std": 2}}
        ],
        "params": {
            "rsi_buy": 30,
            "rsi_sell": 70,
            "bb_candle_lookback": 1  # Do not make this to a big number
        }
    },

    "MA": {  # <--- ny strategi
        "indicators": [
            {"name": "MA", "params": {"period": 20, "ma_type": "SMA"}}
        ],
        "params": {
            "period": 20,  # längd på MA
            "ma_type": "SMA"  # "SMA" eller "EMA"
        }
    }
}
