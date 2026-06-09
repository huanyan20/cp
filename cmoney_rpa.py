import argparse
import json
import os
import time

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# P6: account-config logic extracted to cmoney_client; re-exported for backward compat.
# signal-validation and rebalance-planning were likewise extracted to dedicated modules
# and re-exported here (see __all__) so existing callers of ``cmoney_rpa.<symbol>`` keep
# working. ``_read_execution_log`` / ``_write_execution_log`` are also used internally.
from cmoney_client import get_accounts_config, get_auto_aids
from rebalance_planner import (
    _current_lots_from_rpa,
    build_dry_run_diff,
    build_rebalance_plan,
    run_signal_file,
    write_dry_run_diff,
)
from signal_validator import (
    EXECUTION_LOG_FILE,
    SignalError,
    _normalize_sid,
    _read_execution_log,
    _validate_created_at,
    _write_execution_log,
    load_signal,
    record_signal,
    signal_was_executed,
)
from stock_universe import TICKERS_TECH_EXPANDED

# Backward-compat public surface: names extracted during P6 but still importable as
# ``cmoney_rpa.<symbol>``. Listing them in __all__ marks the re-exports as intentional.
__all__ = [
    "CMoneyRPA",
    "EXECUTION_LOG_FILE",
    "SignalError",
    "TICKERS_TECH_EXPANDED",
    "_current_lots_from_rpa",
    "_normalize_sid",
    "_read_execution_log",
    "_validate_created_at",
    "_write_execution_log",
    "build_dry_run_diff",
    "build_rebalance_plan",
    "get_accounts_config",
    "get_auto_aids",
    "load_signal",
    "record_signal",
    "run_signal_file",
    "signal_was_executed",
    "write_dry_run_diff",
]


class CMoneyRPA:
    def __init__(self, cookie_string=None, aid=None, account_name="Default"):
        self.account_name = account_name
        load_dotenv(override=True)
        self.cookie_string = cookie_string or os.getenv("CMONEY_COOKIE")
        if not self.cookie_string:
            raise ValueError(
                f"[{self.account_name}] 請設定 CMONEY_COOKIE（請從瀏覽器 F12 複製）"
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                "Cookie": self.cookie_string,
            }
        )

        self.aid = aid

    def _get_aid(self):
        """獲取帳戶 ID (aid)"""
        if self.aid:
            return self.aid

        vt_url = "https://www.cmoney.tw/vt/main-page.aspx"
        res = self.session.get(vt_url)
        if "auth.cmoney.tw" in res.url:
            raise PermissionError(
                "[錯誤] Cookie 已經失效或不正確！請重新至瀏覽器登入大富翁，並複製最新的 Cookie 貼入 .env 中。"
            )

        soup = BeautifulSoup(res.text, "html.parser")
        page_data = soup.find(id="PageData")
        if page_data and page_data.get("aid"):
            self.aid = page_data.get("aid")
            print(f"[RPA] [{self.account_name}] 成功抓取 Account ID: {self.aid}")
            return self.aid
        else:
            raise ValueError(
                "無法從 main-page.aspx 抓取到 aid，請確認 Cookie 是否具有大富翁權限。"
            )

    def get_account_status(self):
        print("[RPA] (Requests) 正在獲取大富翁總資產與庫存 (透過 ASHX API)...")
        aid = self._get_aid()

        # 1. 取得總資產
        info_url = (
            f"https://www.cmoney.tw/vt/ashx/accountdata.ashx?act=AccountInfo&aid={aid}"
        )
        res_info = self.session.get(info_url)
        info_data = res_info.json()

        total_assets = float(str(info_data.get("AllAssets", "0")).replace(",", ""))
        available_cash = float(str(info_data.get("Funds", "0")).replace(",", ""))

        # 2. 取得庫存
        inv_url = f"https://www.cmoney.tw/vt/ashx/accountdata.ashx?act=InventoryDetail&aid={aid}"
        res_inv = self.session.get(inv_url)
        inv_data = res_inv.json()

        inventory = {}
        if isinstance(inv_data, list):
            for item in inv_data:
                # 只統計現股買進 (Bs=66 且 TkT='現股')
                if item.get("Bs") == "66" and item.get("TkT") == "現股":
                    ticker = item.get("Id")
                    qty = int(item.get("IQty", 0))
                    if ticker in inventory:
                        inventory[ticker] += qty
                    else:
                        inventory[ticker] = qty

        print(
            f"[RPA] 當前總資產: {total_assets}, 可用現金: {available_cash}, 庫存: {inventory}"
        )
        return {
            "total_assets": total_assets,
            "available_cash": available_cash,
            "inventory": inventory,
        }

    def place_order(self, ticker: str, action: str, quantity: int) -> bool:
        """送出單一委託。

        Returns
        -------
        bool
            True  — API 確認送出成功（status=0, message=""）
            False — 任何錯誤：price=0、API 拒絕、網路異常等
        """
        if quantity <= 0:
            return False

        aid = self._get_aid()

        # 尋找對應的 yfinance ticker 來抓取最新價格
        yf_ticker_str = f"{ticker}.TW"
        for t in TICKERS_TECH_EXPANDED:
            if t.startswith(ticker):
                yf_ticker_str = t
                break

        try:
            curr_df = yf.Ticker(yf_ticker_str).history(period="1d")
            if curr_df.empty:
                raise ValueError("No price data")
            raw_price = float(curr_df["Close"].iloc[-1])

            # 台灣股市跳動單位 (Tick Size) 處理
            if raw_price < 10:
                tick = 0.01
            elif raw_price < 50:
                tick = 0.05
            elif raw_price < 100:
                tick = 0.1
            elif raw_price < 500:
                tick = 0.5
            elif raw_price < 1000:
                tick = 1.0
            else:
                tick = 5.0

            current_price = round(raw_price / tick) * tick

            # 取小數點後兩位，避免過長，如果是整數則不加小數點
            if current_price.is_integer():
                limit_price = str(int(current_price))
            else:
                limit_price = f"{current_price:.2f}"
        except Exception as e:
            print(f"[RPA] 無法獲取 {ticker} 最新股價：{e}")
            limit_price = "0"

        # ── Bug #3 防呆：price=0 時絕對不送單 ────────────────────────────────
        if limit_price == "0":
            print(
                f"[RPA] [{self.account_name}] [錯誤] {ticker} 無法取得有效價格，"
                f"跳過委託（price=0 禁止下單）。請稍後手動確認。"
            )
            return False

        print(
            f"[RPA] [{self.account_name}] 準備委託：{action} {ticker}，數量：{quantity} 張，限價：{limit_price}"
        )

        # 參數映射
        tradekind = "c"  # c: 現股
        bs_type = "b" if action == "BUY" else "s"

        # 呼叫下單 API
        order_url = (
            f"https://www.cmoney.tw/vt/ashx/userset.ashx"
            f"?act=NewEntrust&aid={aid}&stock={ticker}&price={limit_price}"
            f"&ordqty={quantity}&tradekind={tradekind}&type={bs_type}&hasWarrant=false"
        )

        res_order = self.session.get(order_url)
        try:
            result = res_order.json()
            # ── Bug #2 修正：只有 status=0 且 message="" 才算成功 ──────────────
            if result.get("status") == 0 and result.get("message") == "":
                print(
                    f"[RPA] [{self.account_name}] (API) 成功送出委託：{action} {ticker} {quantity} 張"
                )
                time.sleep(0.5)  # 避免觸發限速
                return True
            elif result.get("status") == 0 and result.get("message") == "Bad Request":
                # CMoney VT 有時在資金不足時會回傳 status 0 + Bad Request
                print(
                    f"[RPA] [{self.account_name}] [警告] 下單失敗：可能餘額不足或價格不符 tick size"
                    f"（回傳 Bad Request），{action} {ticker} {quantity} 張 未成功送出。"
                )
            else:
                msg = result.get("message", "未知錯誤")
                print(
                    f"[RPA] [{self.account_name}] [警告] 下單失敗：{msg}，"
                    f"{action} {ticker} {quantity} 張 未成功送出。"
                )
        except Exception:
            print(
                f"[RPA] [{self.account_name}] [錯誤] 解析下單回傳失敗，Raw: {res_order.text}"
            )

        time.sleep(0.5)  # 避免觸發限速（失敗路徑也需等待）
        return False

    def close(self):
        self.session.close()

    def execute_signal(self, signal_path: str, execute: bool, sell_only: bool = False):
        """處理交易訊號。

        Parameters
        ----------
        signal_path : str
            signal.json 檔案路徑
        execute : bool
            True = 真實下單；False = Dry-Run
        sell_only : bool, optional
            True = T+1 模式：只送 SELL 單，BUY 單存為 pending_buys_*.json 等隔日執行。
            False（預設）= 同日完成全部 SELL + BUY（舊行為）。
        """
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"找不到訊號檔案: {signal_path}")

        with open(signal_path, encoding="utf-8") as f:
            signal_data = json.load(f)

        signal_id = signal_data.get("signal_id")
        target_weights = signal_data.get("target_weights", {})

        current_aid = self._get_aid()
        executed_key = f"{signal_id}_{current_aid}"
        sell_done_key = f"{signal_id}_{current_aid}_sell_done"

        # 檢查是否已執行過（完整完成 或 sell-only 模式下的賣單已送出）
        executed_signals = _read_execution_log()

        if executed_key in executed_signals:
            print(f"[RPA] [{self.account_name}] 訊號 {signal_id} 已執行過，略過。")
            return

        if sell_only and sell_done_key in executed_signals:
            print(
                f"[RPA] [{self.account_name}] 訊號 {signal_id} 賣單已送出（等待隔日執行買單），略過。"
            )
            return

        print(
            f"[RPA] [{self.account_name}] 開始處理訊號: {signal_id} "
            f"(Execute: {execute}, SellOnly: {sell_only})"
        )
        account_info = self.get_account_status()
        total_assets = account_info["total_assets"]
        inventory = account_info["inventory"]

        # 獲取當前價格來計算張數

        orders = []

        for vt_ticker, target_weight in target_weights.items():
            if vt_ticker == "CASH":
                continue
                
            if target_weight < 0:
                print(f"[錯誤] 不支援負權重: {vt_ticker}")
                continue

            yf_ticker = vt_ticker
            vt_ticker_id = vt_ticker.split(".")[0]

            try:
                stock = yf.Ticker(yf_ticker)
                hist = stock.history(period="1d")
                price = hist["Close"].iloc[-1] if not hist.empty else 0
            except Exception:
                price = 0

            if price == 0 and target_weight > 0:
                print(f"[警告] {vt_ticker} 無法獲取價格，略過。")
                continue

            target_amt = total_assets * target_weight
            target_lots = int(target_amt / (price * 1000)) if price > 0 else 0
            current_lots = inventory.get(vt_ticker_id, 0)

            diff = target_lots - current_lots
            if diff > 0:
                orders.append(
                    {
                        "ticker": vt_ticker_id,
                        "action": "BUY",
                        "qty": diff,
                        "price": price,
                    }
                )
            elif diff < 0:
                orders.append(
                    {
                        "ticker": vt_ticker_id,
                        "action": "SELL",
                        "qty": -diff,
                        "price": price,
                    }
                )

        # 現有庫存中不在訊號裡面的也要賣出 (歸零)
        # （庫存清倉賣單 price=0 欄位只作佔位，place_order 會即時重抓）
        target_ticker_ids = [t.split(".")[0] for t in target_weights.keys()]
        for inv_ticker, current_lots in inventory.items():
            if inv_ticker not in target_ticker_ids and current_lots > 0:
                orders.append(
                    {
                        "ticker": inv_ticker,
                        "action": "SELL",
                        "qty": current_lots,
                        "price": 0,
                    }
                )

        if not orders:
            print(f"[RPA] [{self.account_name}] 無需調倉。")
            if execute:
                executed_signals.append(executed_key)
                _write_execution_log(executed_signals)
                print(f"[RPA] [{self.account_name}] 訊號 {signal_id} 無需調倉，已標記為完成。")
            return

        sell_orders = [o for o in orders if o["action"] == "SELL"]
        buy_orders  = [o for o in orders if o["action"] == "BUY"]

        # ── Bug #3 防呆：目標調倉賣單若 price==0 代表行情抓取失敗，跳過並警告 ─
        skipped_zero_price = [
            o for o in sell_orders
            if o["price"] == 0 and o["ticker"] in target_ticker_ids
        ]
        if skipped_zero_price:
            for o in skipped_zero_price:
                print(
                    f"[RPA] [{self.account_name}] [警告] SELL {o['ticker']} 因 price=0 被跳過，"
                    f"請手動確認後處理。"
                )
            sell_orders = [o for o in sell_orders if o not in skipped_zero_price]

        # ── Bug #2 修正：追蹤每筆委託的成功 / 失敗 ──────────────────────
        order_results: list[bool] = []

        # 先執行賣單
        for order in sell_orders:
            if execute:
                ok = self.place_order(order["ticker"], order["action"], order["qty"])
                order_results.append(ok)
            else:
                print(
                    f"[Dry-Run] 將委託 {order['action']} {order['ticker']} {order['qty']} 張"
                )

        # ── T+1 模式：賣單送完後，把 BUY 單存入 pending 檔，等隔日執行 ──────
        if sell_only:
            from settings import load_settings as _ls  # noqa: PLC0415
            _pending_dir = _ls().paths.pending_buys_dir
            _pending_dir.mkdir(parents=True, exist_ok=True)
            pending_path = str(_pending_dir / f"pending_buys_{signal_id}_{current_aid}.json")
            if buy_orders:
                pending_data = {
                    "signal_id": signal_id,
                    "aid": current_aid,
                    "created_at": __import__("datetime").datetime.now().isoformat(),
                    "orders": buy_orders,
                }
                if execute:
                    with open(pending_path, "w", encoding="utf-8") as f:
                        json.dump(pending_data, f, indent=4, ensure_ascii=False)
                    print(
                        f"[RPA] [{self.account_name}] {len(buy_orders)} 筆買單已存入 {pending_path}，"
                        f"將於隔日開盤前執行。"
                    )
                else:
                    for o in buy_orders:
                        print(
                            f"[Dry-Run] 將排程 BUY {o['ticker']} {o['qty']} 張 至隔日執行"
                            f"（所需資金 ~{o['qty'] * o['price'] * 1000:,.0f} TWD）"
                        )

            # 標記賣單階段完成
            if execute:
                sell_ok = bool(order_results) and all(order_results) and not skipped_zero_price
                if sell_ok:
                    executed_signals.append(sell_done_key)
                    _write_execution_log(executed_signals)
                    print(
                        f"[RPA] [{self.account_name}] 訊號 {signal_id} 賣單階段已完成，"
                        f"買單排程至隔日。"
                    )
                else:
                    failed_count = order_results.count(False) if order_results else 0
                    print(
                        f"[RPA] [{self.account_name}] [嚴重警告] 訊號 {signal_id} 賣單有 "
                        f"{failed_count} 筆失敗，未標記賣單完成，請手動核查！"
                    )
            return  # sell_only 結束，不繼續執行買單

        # ── 非 T+1 模式（舊行為）：等待撮合後送出買單 ──────────────────────
        if execute and sell_orders:
            print(
                f"[RPA] [{self.account_name}] 已送出賣單，等待 5 秒讓系統撮合釋放資金..."
            )
            time.sleep(5)

        for order in buy_orders:
            if execute:
                ok = self.place_order(order["ticker"], order["action"], order["qty"])
                order_results.append(ok)
            else:
                print(
                    f"[Dry-Run] 將委託 {order['action']} {order['ticker']} {order['qty']} 張"
                )

        # ── Bug #2 修正：只有全部委託成功 + 無 price=0 被跳過，才標記完成 ──
        if execute:
            all_succeeded = bool(order_results) and all(order_results)
            has_skipped = bool(skipped_zero_price)

            if all_succeeded and not has_skipped:
                executed_signals.append(executed_key)
                _write_execution_log(executed_signals)
                print(f"[RPA] [{self.account_name}] 訊號 {signal_id} 已標記為完成。")
            else:
                failed_count = order_results.count(False) if order_results else 0
                skip_msg = (
                    f"、{len(skipped_zero_price)} 筆 price=0 賣單被跳過" if has_skipped else ""
                )
                print(
                    f"[RPA] [{self.account_name}] [嚴重警告] 訊號 {signal_id} 有 {failed_count} 筆委託失敗"
                    f"{skip_msg}。本次訊號【未】標記為完成，下次執行時將重新嘗試。"
                    f"請手動核查委託狀態！"
                )

    def execute_pending_buys(self, execute: bool, max_pending_days: int = 3, half_buys: bool = False):
        """執行前一日存下的 Pending BUY 單（T+1 模式專用）。

        流程：
        1. 掃描目錄下屬於本帳戶的 pending_buys_*_{aid}.json
        2. 檢查建立時間是否超過 max_pending_days（過期則刪除，不執行）
        3. 若 half_buys=True，將所有預計買進數量除以 2 (無條件捨去)。
        4. 取得 available_cash，逐筆 BUY 單驗資金後下單
        5. 全部處理完後刪除 pending 檔，並將訊號標記為完整完成

        Parameters
        ----------
        execute : bool
            True = 真實下單；False = Dry-Run
        max_pending_days : int
            Pending 檔最長存活天數，超過則視為過期自動刪除（預設 3 天）
        half_buys : bool
            True = 執行減半策略 (遇到 WARN 時)
        """
        import datetime
        import glob

        from settings import load_settings as _ls  # noqa: PLC0415
        _pending_dir = _ls().paths.pending_buys_dir
        _pending_dir.mkdir(parents=True, exist_ok=True)

        current_aid = self._get_aid()
        pattern = str(_pending_dir / f"pending_buys_*_{current_aid}.json")
        pending_files = glob.glob(pattern)

        if not pending_files:
            print(f"[RPA] [{self.account_name}] 無 Pending BUY 單需要執行。")
            return

        executed_signals = _read_execution_log()

        for pending_path in pending_files:
            with open(pending_path, encoding="utf-8") as f:
                pending_data = json.load(f)

            signal_id  = pending_data.get("signal_id", "unknown")
            created_at = pending_data.get("created_at", "")
            buy_orders = pending_data.get("orders", [])

            # ── 過期檢查 ──────────────────────────────────────────────────
            try:
                created_dt = datetime.datetime.fromisoformat(created_at)
                age_days = (datetime.datetime.now() - created_dt).days
                if age_days > max_pending_days:
                    print(
                        f"[RPA] [{self.account_name}] [警告] Pending 檔 {pending_path} 已過期 "
                        f"（{age_days} 天），自動刪除，不執行。"
                    )
                    os.remove(pending_path)
                    continue
            except Exception:
                pass  # 無法解析日期則繼續執行

            print(
                f"[RPA] [{self.account_name}] 開始執行 Pending BUY 單：{signal_id} "
                f"（共 {len(buy_orders)} 筆）"
            )

            # ── 取得可用現金 ───────────────────────────────────────────────
            account_info  = self.get_account_status()
            available_cash = account_info["available_cash"]
            simulated_cash = available_cash  # 模擬扣除，依序確認每筆能否負擔

            order_results: list[bool] = []
            skipped_cash: list[dict] = []

            for order in buy_orders:
                ticker = order["ticker"]
                qty    = order["qty"]
                price  = order.get("price", 0)
                
                if half_buys:
                    qty = qty // 2
                    if qty <= 0:
                        print(
                            f"[RPA] [{self.account_name}] [警告] BUY {ticker} 減碼後張數為 0，跳過。"
                        )
                        skipped_cash.append(order)
                        continue
                        
                required = qty * price * 1000

                if price <= 0:
                    print(
                        f"[RPA] [{self.account_name}] [警告] BUY {ticker} price=0，跳過。"
                    )
                    skipped_cash.append(order)
                    continue

                if simulated_cash < required:
                    print(
                        f"[RPA] [{self.account_name}] [警告] BUY {ticker} {qty} 張"
                        f"（需 {required:,.0f} TWD，可用 {simulated_cash:,.0f} TWD），資金不足跳過。"
                    )
                    skipped_cash.append(order)
                    continue

                if execute:
                    ok = self.place_order(ticker, "BUY", qty)
                    order_results.append(ok)
                    if ok:
                        simulated_cash -= required
                else:
                    print(
                        f"[Dry-Run] 將執行 Pending BUY {ticker} {qty} 張"
                        f"（需 {required:,.0f} TWD，可用 {simulated_cash:,.0f} TWD）"
                    )
                    simulated_cash -= required

            # ── 刪除 pending 檔，避免重複執行 ──────────────────────────────
            if execute:
                os.remove(pending_path)
                print(f"[RPA] [{self.account_name}] Pending 檔 {pending_path} 已刪除。")

                # 將訊號升級為完整完成（移除 sell_done key，寫入正式 key）
                sell_done_key = f"{signal_id}_{current_aid}_sell_done"
                executed_key  = f"{signal_id}_{current_aid}"
                if sell_done_key in executed_signals:
                    executed_signals.remove(sell_done_key)
                if executed_key not in executed_signals:
                    executed_signals.append(executed_key)
                _write_execution_log(executed_signals)

                if skipped_cash:
                    print(
                        f"[RPA] [{self.account_name}] [警告] 訊號 {signal_id} 有 "
                        f"{len(skipped_cash)} 筆因資金不足未能買入，已標記完成（避免過期重試）。"
                    )
                else:
                    print(
                        f"[RPA] [{self.account_name}] 訊號 {signal_id} Pending BUY 全部執行完畢，"
                        f"已標記為完整完成。"
                    )


# Backward-compat re-exports (get_accounts_config / get_auto_aids from cmoney_client,
# signal_validator and rebalance_planner symbols) are imported at the top of this module.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMoney RPA 交易執行器")
    parser.add_argument("--signal", type=str, help="訊號 JSON 檔案路徑")
    parser.add_argument(
        "--execute", action="store_true", help="是否真實送出委託 (未指定則為 Dry-Run)"
    )
    parser.add_argument(
        "--sell-only", action="store_true", help="T+1 模式：只送出賣單，買單存為 pending 檔"
    )
    parser.add_argument(
        "--pending-buys", action="store_true", help="T+1 模式：執行前一日存下的 Pending 買單"
    )
    parser.add_argument(
        "--half-buys", action="store_true", help="T+1 模式：執行 Pending 買單時將數量減半"
    )

    args = parser.parse_args()
    print("=== CMoney RPA Multiple Accounts Execution ===")

    accounts_config = get_accounts_config()
    if not accounts_config:
        print("[錯誤] 找不到任何 CMONEY_COOKIE 設定，請檢查 .env 檔案")

    for config in accounts_config:
        print(f"\n--- 正在處理帳戶: {config['name']} ---")
        try:
            rpa = CMoneyRPA(
                cookie_string=config["cookie"],
                aid=config["aid"],
                account_name=config["name"],
            )
            
            if args.pending_buys:
                rpa.execute_pending_buys(args.execute, half_buys=args.half_buys)
            elif args.signal:
                rpa.execute_signal(args.signal, args.execute, sell_only=args.sell_only)
            else:
                print(f"[{config['name']}] 狀態查詢：")
                rpa.get_account_status()
            rpa.close()
        except Exception as e:
            print(f"[錯誤] {config['name']} 執行失敗: {e}")
