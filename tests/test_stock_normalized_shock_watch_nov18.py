from stock_normalized_shock_watch_nov18 import inspect_snapshot
from src.structure_observation_store import frame_hash
from src.stock_structure_observer import NATIVE_SOURCE_SHA


def test_latest_closed_only_boundary_keeps_the_actual_native_scanner_decision():
    bucket=2000000700;times=list(range(bucket-163*900,bucket,900))
    frame=dict(time=times,open=[100.]*163,close=[100.]*163,high=[100.2]*163,
        low=[99.8]*163,volume=[1.]*163)
    rows=[]
    for index in range(12):
        setup=(dict(entry_time=bucket,signal_bar_ts=bucket-900,direction='LONG',
            atr14_fraction=.004,target_r=2.5) if index==0 else None)
        rows.append(dict(symbol=f'S{index}',instrument=f'S{index}-X',quote_ok=False,
            closed_frame=frame,payload=dict(frame=dict(sha256=frame_hash(frame)),native_candidate=setup,
                received_at=bucket+21,book=dict(ok=False))))
    scan=dict(bucket=bucket,status='complete',summary=dict(history_complete=True,
        native_source_sha256=NATIVE_SOURCE_SHA,expected=12,decision_at=bucket+20),observations=rows)
    observed=inspect_snapshot(scan,dict(first_eligible_bucket=bucket))
    assert observed['status']=='observed'
    candidates=observed['strategies']['native_swing2_original_target']
    assert len(candidates)==1 and candidates[0]['symbol']=='S0'
    assert observed['strategies']['normalized_z_reversal_swing2_original_target']==[]
