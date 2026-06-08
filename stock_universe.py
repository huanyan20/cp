"""
stock_universe.py - 定義台股交易池與總體經濟指標
"""

# 45 檔科技股 (Tech30 Expanded Edition)
TICKERS_TECH_EXPANDED = [
    # 半導體 / IC 設計 / 晶圓 / 封測
    "2330.TW",
    "2454.TW",
    "2303.TW",
    "2408.TW",
    "2379.TW",
    "3034.TW",
    "3443.TW",
    "3661.TW",
    "5269.TW",
    "3529.TWO",
    "8299.TWO",
    "5347.TWO",
    "6488.TWO",
    "5483.TWO",
    "6415.TW",
    "8016.TW",
    "3711.TW",
    # AI Server / ODM / PC / 電源
    "2317.TW",
    "2382.TW",
    "3231.TW",
    "2356.TW",
    "6669.TW",
    "2324.TW",
    "2357.TW",
    "2376.TW",
    "2308.TW",
    "6409.TW",
    # 散熱
    "3017.TW",
    "3324.TWO",
    "3653.TW",
    "2421.TW",
    "8996.TW",
    # PCB / 載板 / 材料
    "3037.TW",
    "3189.TW",
    "2368.TW",
    "8046.TW",
    "6213.TW",
    "6274.TWO",
    # 網通 / 連接線
    "2345.TW",
    "3380.TW",
    "6285.TW",
    "3665.TW",
    # 半導體設備
    "3131.TWO",
    "3583.TW",
    "6187.TWO",
]

TICKER_NAMES = {
    "0052.TW": "Yuanta Tech",
    "00881.TW": "Cathay 5G+",
    "00891.TW": "CTBC Semi",
    "00892.TW": "Fubon Semi",
    "2454.TW": "MediaTek",
    "2317.TW": "Foxconn",
    "2382.TW": "Quanta",
    "3711.TW": "ASE Tech",
    "3324.TWO": "Shuang Hong",
    "3017.TW": "AVC",
    "3653.TW": "Jentech",
    "2421.TW": "Sunon",
    "8996.TW": "Kaori",
    "3131.TWO": "Advanced Semica",
    "3583.TW": "Scientech",
    "6187.TWO": "Wanrun",
    "2330.TW": "TSMC",
    "2303.TW": "UMC",
    "2379.TW": "Realtek",
    "3034.TW": "Novatek",
}

MACRO_TICKERS_RL = ["^TWII", "^IXIC", "USDTWD=X"]

MACRO_TICKERS_FLOW = [
    "^VIX",
    "^TNX",
    "BTC-USD",
    "ETH-USD",
    "NQ=F",
    "ES=F",
    "JPY=X",
    "DX-Y.NYB",
]

# Backward-compatible alias for the RL portfolio baseline.
MACRO_TICKERS = MACRO_TICKERS_RL

# ─────────────────────────────────────────────
# 板塊分組（用於計算 sector_flow 資金流向特徵）
# ─────────────────────────────────────────────

SECTOR_GROUPS: dict[str, list[str]] = {
    "semiconductor": [
        "2330.TW", "2454.TW", "2303.TW", "2408.TW", "2379.TW", "3034.TW",
        "3443.TW", "3661.TW", "5269.TW", "3529.TWO", "8299.TWO", "5347.TWO",
        "6488.TWO", "5483.TWO", "6415.TW", "8016.TW", "3711.TW",
    ],
    "ai_server": [
        "2317.TW", "2382.TW", "3231.TW", "2356.TW", "6669.TW",
        "2324.TW", "2357.TW", "2376.TW", "2308.TW", "6409.TW",
    ],
    "cooling": ["3017.TW", "3324.TWO", "3653.TW", "2421.TW", "8996.TW"],
    "pcb": ["3037.TW", "3189.TW", "2368.TW", "8046.TW", "6213.TW", "6274.TWO"],
    "networking": ["2345.TW", "3380.TW", "6285.TW", "3665.TW"],
    "semi_equipment": ["3131.TWO", "3583.TW", "6187.TWO"],
}


def get_ticker_sector(ticker: str) -> str:
    """回傳 ticker 所屬板塊名稱，若未定義則回傳 'unknown'。"""
    for sector, tickers in SECTOR_GROUPS.items():
        if ticker in tickers:
            return sector
    return "unknown"
