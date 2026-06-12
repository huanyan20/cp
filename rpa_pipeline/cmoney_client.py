from __future__ import annotations
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv





def get_auto_aids(cookie_string: str) -> list[str]:

    """Scrape all virtual-trading account IDs from the CMoney main page.



    Parameters

    ----------

    cookie_string:

        The full ``Cookie`` header value from a logged-in browser session.



    Returns

    -------

    list[str]

        Discovered account IDs (``aid``).  Empty list on any failure.

    """

    try:

        res = requests.get(

            "https://www.cmoney.tw/vt/main-page.aspx",

            headers={"User-Agent": "Mozilla/5.0", "Cookie": cookie_string},

            timeout=10,

        )

        soup = BeautifulSoup(res.text, "html.parser")

        links = [

            a["href"]

            for a in soup.find_all("a", href=True)

            if "aid=" in a["href"].lower()

        ]



        aids: list[str] = []

        for link in links:

            match = re.search(r"aid=(\d+)", link, re.IGNORECASE)

            if match:

                aid = match.group(1)

                if aid not in aids:

                    aids.append(aid)



        # Fallback: read PageData element if no hrefs contained an aid

        if not aids:

            page_data = soup.find(id="PageData")

            if page_data and page_data.get("aid"):

                aids.append(page_data.get("aid"))



        return aids

    except Exception as exc:

        print(f"[系統] 無法自動獲取 AIDs: {exc}")

        return []





def get_accounts_config() -> list[dict]:

    """Build the list of CMoney accounts to operate on.



    Reads ``CMONEY_COOKIE`` (and optional ``CMONEY_AID`` / ``CMONEY_AID_N`` /

    ``CMONEY_COOKIE_N``) from the environment (or ``.env`` file).



    Returns

    -------

    list[dict]

        Each entry has keys ``name``, ``cookie``, ``aid``.

    """

    load_dotenv(override=True)

    accounts: list[dict] = []



    default_cookie = os.getenv("CMONEY_COOKIE")

    if not default_cookie:

        return accounts



    # 1. Collect explicitly configured AIDs

    explicit_aids: list[str] = []

    if os.getenv("CMONEY_AID"):

        explicit_aids.append(os.getenv("CMONEY_AID"))  # type: ignore[arg-type]

    for i in range(1, 21):

        if os.getenv(f"CMONEY_AID_{i}"):

            explicit_aids.append(os.getenv(f"CMONEY_AID_{i}"))  # type: ignore[arg-type]



    # 2. Auto-discover if none are specified

    if not explicit_aids:

        print("[系統] .env 未手動指定 AID，將自動登入大富翁抓取所有子帳戶...")

        explicit_aids = get_auto_aids(default_cookie)

        if explicit_aids:

            print(f"[系統] 成功抓取到 {len(explicit_aids)} 個帳戶: {explicit_aids}")

        else:

            print("[系統] 找不到任何帳戶 ID。")



    # 3. Build primary account list

    if not explicit_aids:

        accounts.append(

            {"name": "Default_Account", "cookie": default_cookie, "aid": None}

        )

    else:

        for aid in explicit_aids:

            accounts.append(

                {"name": f"Account_{aid}", "cookie": default_cookie, "aid": aid}

            )



    # 4. Support additional CMoney member logins (CMONEY_COOKIE_1..20)

    for i in range(1, 21):

        extra_cookie = os.getenv(f"CMONEY_COOKIE_{i}")

        if not extra_cookie:

            continue

        print(f"[系統] 偵測到額外的會員 Cookie {i}，自動抓取其子帳戶...")

        sub_aids = get_auto_aids(extra_cookie)

        if not sub_aids:

            accounts.append(

                {"name": f"Extra_Login_{i}", "cookie": extra_cookie, "aid": None}

            )

        else:

            for sub_aid in sub_aids:

                accounts.append(

                    {

                        "name": f"Extra_Login_{i}_{sub_aid}",

                        "cookie": extra_cookie,

                        "aid": sub_aid,

                    }

                )



    return accounts

