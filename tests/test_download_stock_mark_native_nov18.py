import pytest
from download_stock_mark_native_nov18 import parse_history


def test_historical_mark_parser_preserves_missing_volume_and_marks_gaps_without_padding():
    rows=[['900000','100','101','99','100','1'],['2700000','100','101','99','100','1']]
    frame,quality=parse_history(rows,900,3600)
    assert frame['time']==[900,2700] and 'volume' not in frame
    assert quality['gaps']==[(900,2700)]


def test_forming_or_out_of_range_history_cannot_become_a_closed_feature():
    rows=[['900000','100','101','99','100','1'],['1800000','100','101','99','100','0'],
          ['2700000','100','101','99','100','1']]
    frame,_=parse_history(rows,900,2700)
    assert frame['time']==[900]


def test_invalid_ohlc_is_unknown_instead_of_repaired_into_an_attractive_path():
    with pytest.raises(ValueError,match='invalid_mark_ohlc'):
        parse_history([['900000','100','99','98','101','1']],900,1800)
