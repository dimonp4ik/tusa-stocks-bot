from unittest.mock import Mock, patch

from src import earnings_calendar as earnings


def test_failed_fetch_is_short_cached_without_repeated_http_timeouts():
    earnings._cache.clear()
    response = Mock()
    response.json.side_effect = ValueError("bad json")
    with patch.object(earnings.time, "time", return_value=1000), \
            patch.object(earnings.requests, "get", return_value=response) as get:
        assert earnings._reporters_on("2026-10-31") == set()
        assert earnings._reporters_on("2026-10-31") == set()
    assert get.call_count == 1


def test_successful_empty_calendar_uses_normal_cache_ttl():
    earnings._cache.clear()
    response = Mock()
    response.json.return_value = {"data": {"rows": []}}
    with patch.object(earnings.time, "time", return_value=1000), \
            patch.object(earnings.requests, "get", return_value=response) as get:
        assert earnings._reporters_on("2026-10-31") == set()
        assert earnings._reporters_on("2026-10-31") == set()
    assert get.call_count == 1


def test_legacy_two_field_cache_remains_readable_during_hot_reload():
    earnings._cache.clear()
    earnings._cache["2026-10-31"] = (1000, {"AAPL"})
    response = Mock()
    response.json.return_value = {"data": {"rows": []}}
    with patch.object(earnings.time, "time", return_value=1001), \
            patch.object(earnings.requests, "get", return_value=response) as get:
        assert earnings._reporters_on("2026-10-31") == {"AAPL"}
    assert get.call_count == 0
