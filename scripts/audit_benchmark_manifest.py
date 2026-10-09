"""Independent exhaustive post-split audit of a benchmark's saved manifest."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from split_audit import shingles


def audit(rows,threshold=.8):
    if not 0<threshold<=1:raise ValueError('Invalid near-duplicate threshold')
    if any(row.get('split') not in ('train','validation','test') for row in rows):raise ValueError('Invalid split assignment')
    signatures=[shingles(row['prompt']) for row in rows];leaks=[]
    for (i,a),(j,b) in itertools.combinations(enumerate(rows),2):
        if a['split']==b['split']:continue
        x,y=signatures[i],signatures[j]
        similarity=len(x&y)/len(x|y)
        same_trace=a.get('trace_id') is not None and (a.get('source'),a['trace_id'])==(b.get('source'),b.get('trace_id'))
        if similarity>=threshold or same_trace:
            leaks.append({'left':i,'right':j,'jaccard':similarity,'same_trace':same_trace})
    return {'format_version':1,'rows':len(rows),'threshold':threshold,'cross_partition_pairs':len(leaks),'leaks':leaks}


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--threshold',type=float,default=.8)
    p.add_argument('--max-rows',type=int,default=5000);a=p.parse_args()
    path=Path(a.manifest);rows=json.loads(path.read_text())
    if len(rows)>a.max_rows:raise ValueError('Exhaustive audit exceeds row budget; increase --max-rows deliberately')
    report=audit(rows,a.threshold);report['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(report,indent=2));return int(report['cross_partition_pairs']>0)

if __name__=='__main__':sys.exit(main())
