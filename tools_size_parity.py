#!/usr/bin/env python3
"""Does the LIVE position size match the one every backtest figure assumes?

backtest._size_mult_for calls itself "Mirror of the live sizing rules in
src/autotrader.py". Mirrors drift: twice in one day the live book carried a
rule the model did not (kNN sizing), and the model carried rules the live
bot did not (VOLUME_SPIKE, OFF_SESSION). Both were found by reading code,
which is not a check that runs.

This drives the real _open_for_user with the exchange replaced by stubs and
reads the margin it was about to send, then divides out the user's base
margin to recover the live multiplier. The model's mirror is called on the
same setup. A disagreement means reported R does not describe the money.

Run: python tools_size_parity.py [runs]
"""
import os, sys, random, tempfile, importlib

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "size.db"))
import src.db as db          # noqa: E402
db.init_db()
import src.autotrader as at  # noqa: E402
import backtest as bt        # noqa: E402
import config as C           # noqa: E402

_CAP = {"margin": None}


def _install_stubs():
    at._creds_of = lambda u: {"api_key": "k", "api_secret": "s", "passphrase": "p"}
    at._dm = lambda *a, **k: None
    at.at_has_open_position = lambda *a, **k: False
    at.at_set_balance = lambda *a, **k: None
    at._check_threshold_cross = lambda *a, **k: None
    at.at_add_position = lambda *a, **k: 1
    at.okx.get_balance = lambda creds: (True, 10000.0)
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
            "symbol": random.choice(["AAPLUSDT", "TSLAUSDT", "NVDAUSDT", "XAUUSDT"]),
            "direction": random.choice(["LONG", "SHORT"]),
            "entry_price": entry,
            "sl": entry * (1 - random.uniform(0.012, 0.035)),
            "session": random.choice(sessions),
            "trend_1h": random.choice(trends),
            "trend_4h": random.choice(trends),
            "volume_ratio": round(random.uniform(0.8, 6.0), 2),
            "bos_extension_atr": round(random.uniform(0.0, 4.0), 2),
            "vol_atr_pct": round(random.uniform(0.002, 0.020), 4),
            "eff_ratio": round(random.uniform(0.0, 0.9), 3),
        }
        live = _live_mult(sig, user)
        if live is None:
            skipped += 1
            continue
        # The model's mirror. Risk normalisation is live-only BY DESIGN (it is
        # what makes 1R cost the same money at any stop width, which is the
        # assumption every R figure rests on), so divide it back out before
        # comparing -- otherwise this reports a difference that is supposed to
        # be there.
        model = bt._size_mult_for(sig)
        risk_pct = abs(entry - sig["sl"]) / entry
        norm = 1.0
        if C.RISK_NORMALIZED_SIZING and risk_pct > C.RISK_REFERENCE_PCT:
            norm = max(C.RISK_SIZE_MULT_MIN, C.RISK_REFERENCE_PCT / risk_pct)
        # Cap first, normalisation last -- the order both bots now use. The
        # reverse (fold normalisation in, then cap the product) lets the
        # ceiling undo it, which is what the stocks autotrader did until
        # 2026-09-02. This harness reported that as five bot bugs before the
        # formula was fixed to mirror the live order; state the order
        # explicitly rather than assume it.
        expect = min(model, float(C.SIZE_MULT_MAX)) * norm
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
