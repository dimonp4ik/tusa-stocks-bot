"""Reprice frozen RAW SMC candidates with documented base taker costs."""
import json,hashlib
from pathlib import Path
from filter_lab import load,gate,metrics

folder=Path('reports/audit_2026_09_08')
p=folder/'filter_universe_raw.csv';mp=folder/'filter_universe.json'
manifest=json.loads(mp.read_text());old=2*(manifest['arguments']['fee_rate']+manifest['arguments']['slippage_rate'])
new=.0012
rows=load(p)
repriced=[dict(r,net_r=r['net_r']+r['cost_r']-r['cost_r']*new/old,cost_r=r['cost_r']*new/old) for r in rows]
cap=4 if Path.cwd().name=='tusa stocks' else 3
result={'status':'RESEARCH_ONLY','old_roundtrip_cost':old,'new_roundtrip_cost':new,
 'old':metrics(gate(rows,cap)),'new':metrics(gate(repriced,cap)),
 'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'source':'https://www.okx.com/en-gb/help/okx-x-perps-eea-fees-overview',
 'limitations':['Same historical candidate generation and fills; only estimated cost changed. No account-specific fee confirmation.',
 'Cost normalization uses original entry notional as in the underlying replay.',
 'Funding remains excluded; new stress metric adds another 10bps to 12bps total baseline.',
 'Previously inspected SWAP data, not actual X-Perp account PnL.']}
(folder/'base_taker_cost_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
