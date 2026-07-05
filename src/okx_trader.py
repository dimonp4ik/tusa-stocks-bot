"""
OKX EU private trade API client (v5) for user autotrading.

Per-user credentials are passed into every call — this module holds no global
API keys. Host defaults to my.okx.com (OKX EU / MiCA entity, where X-Perp
instruments live); override with OKX_TRADE_BASE_URL.

Execution model (mirrors the signal engine):
  - entry: market order, isolated margin, fixed 10x leverage
  - protection: separate reduce-only OCO algo (TP at tp2, SL at sl), market close
  - post-TP1 trailing: the engine amends the algo's SL trigger every monitor
    cycle; engine transitions (TP2_HIT / TP1_TRAIL / EXPIRED...) market-close
    whatever the exchange algo hasn't closed yet.

Every public function returns (ok: bool, payload) and never raises — callers
loop over many users and one user's failure must not break the rest.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone

import requests

_log = logging.getLogger(__name__)

_BASE_URL_DEFAULT = "https://my.okx.com"


def _base_url() -> str:
    return os.getenv("OKX_TRADE_BASE_URL", _BASE_URL_DEFAULT).strip().rstrip("/")


def _timestamp() -> str:
    # OKX wants ISO8601 with milliseconds, e.g. 2026-07-05T12:31:04.123Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sign(secret: str, ts: str, method: str, path: str, body: str) -> str:
    msg = f"{ts}{method}{path}{body}"
    mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _request(creds: dict, method: str, path: str, params: dict = None,
             body: dict = None, timeout: int = 15) -> tuple:
    """Signed request. creds = {api_key, api_secret, passphrase}.
    Returns (ok, data-list-or-error-string)."""
    try:
        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        body_str = json.dumps(body) if body else ""
        ts = _timestamp()
        sign_path = path + query
        headers = {
            "OK-ACCESS-KEY":        creds["api_key"],
            "OK-ACCESS-SIGN":       _sign(creds["api_secret"], ts, method, sign_path, body_str),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": creds["passphrase"],
            "Content-Type":         "application/json",
        }
        url = _base_url() + sign_path
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, headers=headers, data=body_str, timeout=timeout)
        j = resp.json()
        if str(j.get("code")) == "0":
            return True, j.get("data", [])
        # Per-order errors sit inside data[0].sMsg with outer code 1/2
        detail = ""
        try:
            d0 = (j.get("data") or [{}])[0]
            detail = d0.get("sMsg") or ""
        except Exception:
            pass
        return False, f"{j.get('msg') or detail or 'unknown error'} (code {j.get('code')}{', ' + detail if detail and j.get('msg') else ''})"
    except Exception as e:
        return False, f"request failed: {e}"


# ── Public data (no auth) ─────────────────────────────────────────────────────

_spec_cache = {"at": 0.0, "by_id": {}}
_SPEC_TTL = 24 * 3600


def get_xperp_spec(inst_id: str) -> dict | None:
    """Instrument spec {ctVal, lotSz, minSz, tickSz, lever} for an X-Perp.
    Cached 24h (whole X-Perp list in one call)."""
    now = time.time()
    if now - _spec_cache["at"] > _SPEC_TTL or not _spec_cache["by_id"]:
        try:
            resp = requests.get(
                _base_url() + "/api/v5/public/instruments",
                params={"instType": "FUTURES"}, timeout=15,
            )
            data = resp.json().get("data", [])
            by_id = {}
            for x in data:
                iid = str(x.get("instId", ""))
                if "_UM_XPERP-" in iid and x.get("state") == "live":
                    by_id[iid] = {
                        "ctVal":  float(x["ctVal"]),
                        "lotSz":  float(x["lotSz"]),
                        "minSz":  float(x["minSz"]),
                        "tickSz": float(x["tickSz"]),
                        "lever":  float(x.get("lever") or 10),
                    }
            if by_id:
                _spec_cache["by_id"] = by_id
                _spec_cache["at"] = now
        except Exception as e:
            _log.warning(f"get_xperp_spec fetch failed: {e}")
    return _spec_cache["by_id"].get(inst_id)


def get_last_price(inst_id: str) -> float | None:
    try:
        resp = requests.get(
            _base_url() + "/api/v5/market/ticker",
            params={"instId": inst_id}, timeout=10,
        )
        data = resp.json().get("data", [])
        return float(data[0]["last"]) if data else None
    except Exception as e:
        _log.warning(f"get_last_price {inst_id} failed: {e}")
        return None


def round_to_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return px
    steps = round(px / tick)
    # Format through the tick's own decimals to kill float noise (0.30000000004)
    dec = max(0, f"{tick:.10f}".rstrip("0")[::-1].find("."))
    return round(steps * tick, dec)


# ── Account ───────────────────────────────────────────────────────────────────

def get_balance(creds: dict) -> tuple:
    """(ok, total_equity_usd | error). Used for both key validation and sizing."""
    ok, data = _request(creds, "GET", "/api/v5/account/balance")
    if not ok:
        return False, data
    try:
        total = float(data[0].get("totalEq") or 0.0)
        return True, total
    except Exception as e:
        return False, f"balance parse failed: {e}"


def set_leverage(creds: dict, inst_id: str, lever: int = 10) -> tuple:
    """Isolated 10x on the instrument. 59107/'leverage not modified' → fine."""
    ok, data = _request(creds, "POST", "/api/v5/account/set-leverage", body={
        "instId": inst_id, "lever": str(lever), "mgnMode": "isolated",
    })
    if not ok and "not modified" in str(data).lower():
        return True, data
    return ok, data


# ── Orders ────────────────────────────────────────────────────────────────────

def place_market_entry(creds: dict, inst_id: str, direction: str, sz: float) -> tuple:
    """Market entry, isolated margin, net (one-way) position mode.
    direction LONG→buy, SHORT→sell. Returns (ok, ordId | error)."""
    side = "buy" if str(direction).upper() == "LONG" else "sell"
    ok, data = _request(creds, "POST", "/api/v5/trade/order", body={
        "instId":  inst_id,
        "tdMode":  "isolated",
        "side":    side,
        "ordType": "market",
        "sz":      _fmt_sz(sz),
    })
    if not ok:
        return False, data
    try:
        return True, data[0]["ordId"]
    except Exception:
        return True, ""


def place_protection_oco(creds: dict, inst_id: str, direction: str,
                         sl_px: float, tp_px: float) -> tuple:
    """Reduce-only OCO covering the WHOLE position (closeFraction=1):
    SL at engine's stop, TP at tp2 — both market on trigger. The engine
    amends the SL trigger as the post-TP1 trail moves. Returns (ok, algoId|err)."""
    close_side = "sell" if str(direction).upper() == "LONG" else "buy"
    ok, data = _request(creds, "POST", "/api/v5/trade/order-algo", body={
        "instId":        inst_id,
        "tdMode":        "isolated",
        "side":          close_side,
        "ordType":       "oco",
        "reduceOnly":    "true",
        "closeFraction": "1",
        "slTriggerPx":   str(sl_px),
        "slOrdPx":       "-1",     # market on trigger
        "tpTriggerPx":   str(tp_px),
        "tpOrdPx":       "-1",
    })
    if not ok:
        return False, data
    try:
        return True, data[0]["algoId"]
    except Exception:
        return True, ""


def amend_protection_sl(creds: dict, inst_id: str, algo_id: str, new_sl_px: float) -> tuple:
    """Move the OCO's SL trigger (trailing after TP1)."""
    return _request(creds, "POST", "/api/v5/trade/amend-algos", body={
        "instId":         inst_id,
        "algoId":         algo_id,
        "newSlTriggerPx": str(new_sl_px),
        "newSlOrdPx":     "-1",
    })


def cancel_protection(creds: dict, inst_id: str, algo_id: str) -> tuple:
    ok, data = _request(creds, "POST", "/api/v5/trade/cancel-algos",
                        body=[{"instId": inst_id, "algoId": algo_id}])
    # Already triggered/cancelled → treat as done
    if not ok and any(s in str(data).lower() for s in ("not exist", "canceled", "cancelled", "state")):
        return True, data
    return ok, data


def close_position_market(creds: dict, inst_id: str) -> tuple:
    """Market-close the whole isolated position. 'no position' → already flat, ok."""
    ok, data = _request(creds, "POST", "/api/v5/trade/close-position", body={
        "instId": inst_id, "mgnMode": "isolated",
    })
    if not ok and any(s in str(data).lower() for s in ("position", "51023", "51169")):
        # 51023 position not exist / 51169 no position to close — algo got there first
        return True, "already flat"
    return ok, data


def _fmt_sz(sz: float) -> str:
    """Contracts as clean string: 3.0 → '3', 0.5 → '0.5'."""
    s = f"{sz:.8f}".rstrip("0").rstrip(".")
    return s or "0"


def calc_contracts(margin_usd: float, leverage: float, price: float, spec: dict) -> float:
    """margin*leverage → notional → contracts, floored to lotSz. 0 if below minSz."""
    if price <= 0 or not spec:
        return 0.0
    notional = margin_usd * leverage
    raw = notional / (spec["ctVal"] * price)
    lot = spec["lotSz"] or 1.0
    sz = int(raw / lot) * lot
    if sz < spec["minSz"]:
        return 0.0
    return sz
