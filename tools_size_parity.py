#!/usr/bin/env python3
"""Verify that live orders use exactly the client's percentage of balance.

This drives the real _open_for_user with the exchange replaced by stubs and
captures the margin passed to contract sizing. Every setup must keep a 1.0
multiplier: strategy fields may filter entries, but cannot resize them.

Run: python tools_size_parity.py [runs]
"""
import os, sys, random, tempfile

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "size.db"))
import src.db as db          # noqa: E402
db.init_db()
import src.autotrader as at  # noqa: E402

_CAP = {"margin": None}


def _install_stubs():
    at._creds_of = lambda u: {"api_key": "k", "api_secret": "s", "passphrase": "p"}
    at._dm = lambda *a, **k: None
    at.at_has_open_position = lambda *a, **k: False
    at.at_set_balance = lambda *a, **k: None
    at.get_bot_state = lambda *a, **k: '{}'
    at.set_bot_state = lambda *a, **k: None
    at.set_signal_size_mult = lambda *a, **k: None
    at.okx.get_balance = lambda creds: (True, 10000.0)
    at.okx.get_position_size = lambda *a, **k: (True, 0.0)
    at.okx.get_xperp_spec = lambda inst: {"ctVal": 1.0, "lotSz": 0.01,
                                          "minSz": 0.01, "tickSz": 0.01, "lever": 10}
    at.okx.get_last_price = lambda inst: 100.0

    def _calc(margin, lev, px, spec):
        _CAP["margin"] = margin
        return 0.0          # returning 0 makes the caller bail out safely
    at.okx.calc_contracts = _calc


def _live_mult(sig, user):
    _CAP["margin"] = None
    at._open_for_user(user, sig, "TEST-USDT-SWAP", "TEST")
    if _CAP["margin"] is None:
        return None
    base = at._margin_for(user, 10000.0)
    return _CAP["margin"] / base if base else None


def main_() -> int:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    random.seed(23)
    _install_stubs()
    # _margin_for reads size_mode/size_value, not risk_pct. With those absent
    # the base margin is 0, the caller bails at "margin <= 0", and the probe
    # silently compares nothing -- which is exactly how this harness first
    # reported "no comparable pairs".
    user = {"user_id": 1, "size_mode": "percent", "size_value": 2.0,
            "active": 1, "allowed": 1}

    sessions = ["OPEN", "MIDDAY", "CLOSE", "OFF"]
    trends = ["bullish", "bearish", "neutral"]
    agree = dis = skipped = 0
    worst = []
    for _ in range(runs):
        entry = 100.0
        sig = {
            "id": 1,
            "symbol": random.choice(["AAPLUSDT", "TSLAUSDT", "NVDAUSDT", "XAUUSDT"]),
            "direction": random.choice(["LONG", "SHORT"]),
            "entry_price": entry,
            "sl": entry * (1 - random.uniform(0.012, 0.035)),
            "tp1": entry * 1.1,
            "session": random.choice(sessions),
            "trend_1h": random.choice(trends),
            "trend_4h": random.choice(trends),
            "volume_ratio": round(random.uniform(0.8, 6.0), 2),
            "bos_extension_atr": round(random.uniform(0.0, 4.0), 2),
            "vol_atr_pct": round(random.uniform(0.002, 0.020), 4),
            # RSI_STRETCH_SIZE_MULT keys on this. Without it the generated
            # rows never reach the branch and parity passes while proving
            # nothing about it — the exact failure this harness exists to
            # prevent. Range straddles the 68 threshold on purpose.
            "rsi": round(random.uniform(30.0, 80.0), 2),
            "eff_ratio": round(random.uniform(0.0, 0.9), 3),
        }
        if sig["direction"] == "SHORT":
            sig["sl"], sig["tp1"] = entry * 1.03, entry * 0.9
        live = _live_mult(sig, user)
        if live is None:
            skipped += 1
            continue
        expect = 1.0
        if abs(live - expect) < 1e-6:
            agree += 1
        else:
            dis += 1
            if len(worst) < 5:
                worst.append((sig["session"], sig["symbol"], sig["trend_1h"],
                              round(live, 4), round(expect, 4)))
    total = agree + dis
    if total == 0:
        print("АВАРИЯ: ни одной сравнимой пары — стенд не отработал")
        return 1
    print(f"сравнено сетапов: {total}  (пропущено {skipped})")
    print(f"  совпало:   {agree} ({100*agree/total:.1f}%)")
    print(f"  разошлось: {dis} ({100*dis/total:.1f}%)")
    for w in worst:
        print(f"    сессия={w[0]:9s} {w[1]:9s} 1ч={w[2]:8s} живьём={w[3]} модель={w[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
