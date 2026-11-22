from decimal import Decimal
import os
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

import main
from src import okx_closed_pnl as pnl


NOW = 1_795_312_000_000
INSTRUMENT = "LINK-USD_UM_XPERP-310613"


def position(index, *, pnl_value, currency="USDC", kind="2", instrument=INSTRUMENT):
    return dict(instId=instrument, instType="FUTURES", type=kind, ccy=currency,
                cTime=str(NOW - 100_000 - index*200_000), uTime=str(NOW - 10_000 - index*200_000),
                mgnMode="isolated", posId="reused-id", posSide="net", direction="long",
                openMaxPos="2", closeTotalPos="2", realizedPnl=pnl_value)


def trade(number, instrument, timestamp, *, kind, size, order, client=""):
    return dict(instType="FUTURES", instId=instrument, posSide="net", billId=str(number),
                fillTime=str(timestamp), subType=kind, fillSz=size, ordId=order, clOrdId=client,
                side="buy" if kind=="3" else "sell")


def lifecycle(row, client, high):
    instrument, opened, closed = row["instId"], int(row["cTime"]), int(row["uTime"])
    order = str(high)
    return [trade(high+3, instrument, closed, kind="5", size="1.5", order=order+"sl"),
            trade(high+2, instrument, closed-2000, kind="5", size="0.5", order=order+"tp"),
            trade(high+1, instrument, opened+1, kind="3", size="1", order=order, client=client),
            trade(high, instrument, opened, kind="3", size="1", order=order, client=client)]


def test_repeated_position_id_and_multiple_closes_are_not_counted_twice():
    ours = position(0, pnl_value="-0.755")
    peer = position(1, pnl_value="2.0001", currency="USD")
    partial = position(2, pnl_value="3000", kind="1")

    def read(path, params):
        assert params["instType"] == "FUTURES"
        return True, [ours, peer, partial] if path == pnl._HISTORY else lifecycle(ours, "local", 101)

    report = pnl.reconcile(read, {"local": (INSTRUMENT, "LONG")}, now_ms=NOW)
    assert report["account"] == {"USDC": Decimal("-0.755"), "USD": Decimal("2.0001")}
    assert report["bot"] == {"USDC": Decimal("-0.755")}
    assert report["positions"] == 2 and report["matched"] == 1 and report["partial"] == 1
    assert "-0.7550 USDC" in pnl.format_closed_pnl(report)


def test_mixed_entry_identity_preserves_account_only():
    row = position(0, pnl_value="0.90")
    fills = lifecycle(row, "local", 101)
    fills[-1]["clOrdId"] = "external"
    report = pnl.reconcile(lambda p, _: (True, [row] if p == pnl._HISTORY else fills),
                           {"local": (INSTRUMENT, "LONG")}, now_ms=NOW)
    assert report["account"] == {"USDC": Decimal("0.90")}
    assert not report["matched"] and not report["bot"]


def test_pagination_reads_older_rows_and_rejects_a_nonadvancing_cursor():
    one = position(0, pnl_value="1")
    two = position(1, pnl_value="2")
    calls = []

    def read(path, params):
        calls.append(params.get("after"))
        return True, {None: [one], one["uTime"]: [two], two["uTime"]: []}[calls[-1]]

    with patch.object(pnl, "_PAGE", 1):
        report = pnl.reconcile(read, {}, now_ms=NOW)
        assert report["account"] == {"USDC": Decimal("3")}
        assert calls == [None, one["uTime"], two["uTime"]]
        assert pnl.reconcile(lambda *_: (True, [one]), {}, now_ms=NOW) is None


def test_missing_exchange_fills_keep_signed_account_data_but_no_bot_value():
    row = position(0, pnl_value="-3")
    report = pnl.reconcile(lambda p, _: (True, [row]) if p==pnl._HISTORY else (False, "network"),
                           {"local": (INSTRUMENT, "LONG")}, now_ms=NOW)
    assert report["account"] == {"USDC": Decimal("-3")}
    assert not report["attribution_available"]
    assert "пока не подтверждён" in pnl.format_closed_pnl(report)


def test_short_liquidation_and_fills_pagination_keep_the_actual_loss():
    row = dict(position(0, pnl_value="-2.41", kind="3"), direction="short")
    opened, closed = int(row["cTime"]), int(row["uTime"])
    fill_one = dict(trade(42, INSTRUMENT, opened, kind="4", size="2",
                          order="short", client="local"), side="sell")
    fill_two = dict(trade(43, INSTRUMENT, closed, kind="105", size="2",
                          order="liquidation"), side="buy")
    fill_calls = []

    def read(path, params):
        if path == pnl._HISTORY:
            return True, [row] if not params.get("after") else []
        fill_calls.append(params.get("after"))
        return True, {None: [fill_two], "43": [fill_one], "42": []}[fill_calls[-1]]

    with patch.object(pnl, "_PAGE", 1):
        report = pnl.reconcile(read, {"local": (INSTRUMENT, "SHORT")}, now_ms=NOW)
    assert fill_calls == [None, "43", "42"]
    assert report["matched"] == 1 and report["bot"] == {"USDC": Decimal("-2.41")}


def test_real_results_button_and_worker_are_private():
    user = dict(allowed=True, api_key_enc="encrypted-place-holder")
    with patch.object(main, "at_get", return_value=user), \
         patch.object(main, "_paper_results_text", return_value="paper"), \
         patch.object(main, "_at_reply_kb") as keyboard, \
         patch.object(main, "_reply") as reply, \
         patch.object(main, "_answer_callback") as ack, \
         patch.object(pnl, "start_user_report", return_value=True) as start:
        main._send_paper_results(-20, 20)
        reply.assert_called_once_with(-20, "paper")
        keyboard.assert_not_called()
        main._at_handle_callback("cb", -20, 20, "at_closed_results")
        start.assert_not_called()
        main._send_paper_results(20, 20)
        assert keyboard.call_args.args[2][0][0]["callback_data"] == "at_closed_results"
        assert main._at_handle_callback("cb", 20, 20, "at_closed_results")
        assert start.call_args.args[:2] == (20, user)
        ack.assert_called_with("cb", "Проверяю историю OKX.")
