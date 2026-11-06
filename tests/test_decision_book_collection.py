from types import SimpleNamespace
from unittest.mock import patch

import main


def test_paper_candidate_collects_labeled_decision_book():
    analysis = {
        "symbol": "AAVEUSDT", "_signal_id": 42, "_shadow_only": True,
    }
    snapshot = SimpleNamespace(
        spread_bps=1.25, depth_usd=25_000.0, imbalance=.15)
    with patch.object(
            main, "get_xperp_instruments",
            return_value={"AAVE": "AAVE-USDT-SWAP"}), patch.object(
            main._okx_trade, "get_xperp_spec",
            return_value={"ctVal": .1}), patch.object(
            main._okx_trade, "get_order_book",
            return_value={"asks": [], "bids": []}), patch.object(
            main, "order_book_snapshot", return_value=snapshot), patch.object(
            main, "record_setup_decision_book", return_value=True) as record:
        main._record_candidate_decision_books([(analysis, 7)])
    record.assert_called_once_with(
        7, spread_bps=1.25, depth_usd=25_000.0, imbalance=.15)


def test_live_published_candidate_collects_fallback_decision_book():
    analysis = {
        "symbol": "AAVEUSDT", "_signal_id": 42, "_shadow_only": False,
    }
    snapshot = SimpleNamespace(
        spread_bps=1.5, depth_usd=12_000.0, imbalance=-.2)
    with patch.object(
            main, "get_xperp_instruments",
            return_value={"AAVE": "AAVE-USDT-SWAP"}), patch.object(
            main._okx_trade, "get_xperp_spec",
            return_value={"ctVal": .1}), patch.object(
            main._okx_trade, "get_order_book",
            return_value={"asks": [], "bids": []}), patch.object(
            main, "order_book_snapshot", return_value=snapshot), patch.object(
            main, "record_setup_decision_book", return_value=True) as record:
        main._record_candidate_decision_books([(analysis, 7)])
    record.assert_called_once_with(
        7, spread_bps=1.5, depth_usd=12_000.0, imbalance=-.2)
