"""Download immutable public-market windows; never print the audit credential."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

BASE='https://web-production-69f0f.up.railway.app/internal/structure-observer/v1/'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('unexpected_export_redirect')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-bucket',type=int,default=0)
    parser.add_argument('--limit',type=int,default=4)
    parser.add_argument('--context',action='store_true',help='Export the separate mark/trade context profile')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    args.output=args.output or Path('reports/mark_context_prospective_nov18' if args.context
        else 'reports/structure_observer_prospective_nov18')
    if args.after_bucket<0 or args.after_bucket%900 or not 1<=args.limit<=4:
        raise ValueError('invalid_export_bounds')
    token=Path('.git/structure_observer_audit.token').read_text(encoding='utf-8').strip()
    request=urllib.request.Request(BASE+('context-export' if args.context else 'export'),method='POST',
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},
        data=json.dumps(dict(after_bucket=args.after_bucket,limit=args.limit)).encode())
    with urllib.request.build_opener(NoRedirect).open(request,timeout=30) as response:
        exported=json.load(response)
    if exported.get('ok') is not True:raise ValueError('export_not_confirmed')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for scan in exported['export']['scans']:
        if scan['completed_at'] is None:
            print(json.dumps(dict(bucket=scan['bucket'],status='unfinished_not_archived')))
            continue
        raw=json.dumps(scan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        path=args.output/f"bucket_{scan['bucket']}.json"
        if path.exists() and path.read_bytes()!=raw:raise ValueError('immutable_export_disagrees')
        if not path.exists():path.write_bytes(raw)
        rows=scan['observations']
        item=dict(bucket=scan['bucket'],status=scan['status'],path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            observations=len(rows),valid_books=sum(bool(r['quote_ok']) for r in rows),
            reconstructed_frames=sum('closed_frame' in r for r in rows))
        if args.context:item['reconstructed_trade_frames']=sum('trade_closed_frame' in r for r in rows)
        manifest.append(item);print(json.dumps(item),flush=True)
    # Each export has its own receipt manifest; earlier raw windows stay fixed.
    target=args.output/f"receipt_after_{args.after_bucket}.json"
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
