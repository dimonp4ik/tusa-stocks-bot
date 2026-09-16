"""Report monthly net outcomes without selecting or fitting new filters."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from filter_lab import load,metrics
folder=Path('reports/audit_2026_09_08')
p=folder/'all_structure_timing.csv'
rows=load(p);months={}
for row in rows:
    month=datetime.fromtimestamp(row['exit_time'],timezone.utc).strftime('%Y-%m')
    months.setdefault(month,[]).append(row)
report={'status':'RESEARCH_ONLY','csv_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'all':metrics(rows),'by_exit_month':{m:metrics(rs) for m,rs in sorted(months.items())},
        'limitations':['Closed-trade R, not account returns or intratrade drawdown.',
        'Already inspected SWAP history; actual fills and funding excluded.',
        'No new filter selected using these monthly outcomes.']}
(folder/'all_structure_monthly.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
