from unittest.mock import Mock, patch

import pytest
import requests

from src import earnings_calendar as earnings


def nasdaq_response(rows, *, provider_code=200, day="Sat, Oct 31, 2026"):
    response = Mock()
    response.json.return_value = {
        "status": {"rCode": provider_code},
        "data": {"asOf": day, "rows": rows},
    }
    return response


def test_failed_fetch_is_short_cached_without_repeated_http_timeouts():
    earnings._cache.clear()
    response = Mock()
    response.json.side_effect = ValueError("bad json")
    with patch.object(earnings.time, "time", return_value=1000), \
            patch.object(earnings.requests, "get", return_value=response) as get:
        assert earnings._reporters_on("2026-10-31") == set()
        assert earnings._reporters_on("2026-10-31") == set()
    assert get.call_count == 1


@pytest.mark.parametrize("rows", [None, []])
def test_successful_empty_calendar_uses_normal_cache_ttl(rows):
    earnings._cache.clear()
    response = nasdaq_response(rows)
    with patch.object(earnings.time, "time", side_effect=[1000, 1000 + earnings._FAILURE_TTL + 1]), \
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


@pytest.mark.parametrize("failure", ["http", "provider", "wrong_day", "bad_rows", "missing_rows"])
def test_calendar_failure_preserves_previous_reporters_and_retries_after_short_ttl(failure):
    day = "2026-10-31"
    start = 1000
    now = start + earnings._CACHE_TTL + 1
    earnings._cache.clear()
    earnings._cache[day] = (start, {"AAPL"}, True)
    rejected = nasdaq_response(None)
    if failure == "http":
        rejected.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    elif failure == "provider":
        rejected = nasdaq_response(None, provider_code=500)
    elif failure == "wrong_day":
        rejected = nasdaq_response([], day="Sun, Nov 1, 2026")
    elif failure == "bad_rows":
        rejected = nasdaq_response([{"symbol": "NVDA"}, {"symbol": ""}])
    else:
        del rejected.json.return_value["data"]["rows"]
    valid = nasdaq_response([{"symbol": "nvda"}])
    clock = [now]
    with patch.object(earnings.time, "time", side_effect=lambda: clock[0]), \
            patch.object(earnings.requests, "get", side_effect=[rejected, valid]) as get:
        assert earnings._reporters_on(day) == {"AAPL"}
        assert earnings._cache[day] == (now, {"AAPL"}, False)
        clock[0] += earnings._FAILURE_TTL - 1
        assert earnings._reporters_on(day) == {"AAPL"}
        get.assert_called_once()
        clock[0] += 2
        assert earnings._reporters_on(day) == {"NVDA"}
        assert get.call_count == 2
