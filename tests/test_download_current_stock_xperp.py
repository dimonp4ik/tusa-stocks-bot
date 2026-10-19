import download_current_stock_xperp as downloader


def test_snapshot_anchor_is_current_interval_open():
    assert downloader.snapshot_anchor_ms(901) == 900_000
    assert downloader.snapshot_anchor_ms(1799.999) == 900_000
    assert downloader.snapshot_anchor_ms(1800) == 1_800_000


def test_snapshot_anchor_never_points_into_future():
    now = 123_456.789
    anchor_seconds = downloader.snapshot_anchor_ms(now) / 1000
    assert anchor_seconds <= now
    assert now - anchor_seconds < downloader.BAR_SECONDS
