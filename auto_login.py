import os
import time

from dotenv import load_dotenv, set_key
from playwright.sync_api import sync_playwright


def login_and_get_cookie():
    # 讀取 .env 中的帳號密碼
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path, override=True)

    username = os.getenv("CMONEY_USERNAME")
    password = os.getenv("CMONEY_PASSWORD")

    if not username or not password:
        print("[錯誤] 找不到 CMONEY_USERNAME 或 CMONEY_PASSWORD，請檢查 .env 設定。")
        return False

    print("[自動登入] 準備啟動瀏覽器...")
    state_path = os.path.join(os.path.dirname(__file__), "playwright_state.json")
    has_state = os.path.exists(state_path)

    # 若沒有狀態檔，代表需要初次登入並人工解 2FA，所以關閉無頭模式
    headless_mode = has_state

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless_mode)

        if has_state:
            print("[自動登入] 偵測到已儲存的裝置狀態，背景登入中...")
            context = browser.new_context(
                storage_state=state_path,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
        else:
            print("[自動登入] 初次登入，將開啟實體視窗，請準備完成雙重認證...")
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

        page = context.new_page()

        try:
            print("[自動登入] 正在前往 CMoney 登入頁面...")
            page.goto("https://www.cmoney.tw/member/login/", timeout=30000)

            # 嘗試填寫表單 (若 storage_state 已經記住登入，可能不用填)
            try:
                page.wait_for_selector("#Account", timeout=5000)
                page.fill("#Account", username)
                page.fill("#Password", password)

                if page.locator("#RememberMe").is_visible():
                    page.check("#RememberMe")

                print("[自動登入] 送出帳號密碼...")
                page.click("#Login")
            except Exception:
                print("[自動登入] 偵測不到登入表單，可能已經處於登入狀態。")

            if not has_state:
                print("\n=======================================================")
                print("[需要人工介入]")
                print("瀏覽器已經彈出！如果您看見雙重認證 (2FA) 畫面，")
                print("請手動輸入簡訊或 Email 驗證碼。")
                print("程式會在此靜候，直到偵測到成功登入為止 (最多等待 90 秒)...")
                print("=======================================================\n")

            # 輪詢等待 cm_at 出現
            max_wait = 90 if not has_state else 15
            for _i in range(max_wait):
                time.sleep(1)
                cookies = context.cookies()
                cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
                if "cm_at" in cookie_str:
                    break

            if "cm_at" not in cookie_str:
                print(
                    "[警告] 等待超時！未取得 cm_at (授權憑證)。可能是 2FA 未完成或被阻擋。"
                )
                if headless_mode:
                    page.screenshot(path="login_failed_screenshot.png")
                    print("[自動登入] 已儲存失敗截圖至 login_failed_screenshot.png")
                return False

            # 登入成功，儲存裝置狀態
            print("[自動登入] 登入成功！已儲存裝置認證狀態 (playwright_state.json)。")
            context.storage_state(path=state_path)

            # 將新 Cookie 寫回 .env
            set_key(env_path, "CMONEY_COOKIE", cookie_str, quote_mode="never")
            print("[自動登入] 成功將最新 Cookie 儲存至 .env！")
            return True

        except Exception as e:
            print(f"[自動登入] 發生錯誤: {e}")
            return False
        finally:
            browser.close()


if __name__ == "__main__":
    success = login_and_get_cookie()
    if not success:
        exit(1)
