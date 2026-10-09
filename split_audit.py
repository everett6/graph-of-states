"""Deterministic connected-component splitting with exhaustive shingle auditing."""
from collections import defaultdict
import hashlib
import json
import re
import unicodedata


def canonical(text):
    return ' '.join(re.findall(r'\w+|[^\w\s]',unicodedata.normalize('NFKC',text).casefold()))


def shingles(text):
    words=canonical(text).split()
    size=min(3,len(words))
    return {tuple(words[index:index+size]) for index in range(len(words)-size+1)} or {()}


def grouped_split(rows,seed=17,threshold=0.8):
    if not 0<threshold<=1: raise ValueError('Near-duplicate threshold must be (0,1]')
    parent=list(range(len(rows)))
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    inverted=defaultdict(list);signatures=[];matches=[];traces={}
    for index,row in enumerate(rows):
        trace=row.get('trace_id')
        if trace is not None:
            trace=(row.get('source',''),str(trace))
            if trace in traces: union(index,traces[trace])
            else: traces[trace]=index
        signature=shingles(row['prompt']);signatures.append(signature)
        candidates=set(other for item in signature for other in inverted[item])
        for other in sorted(candidates):
            previous=signatures[other]
            if min(len(signature),len(previous))/max(len(signature),len(previous)) < threshold: continue
            similarity=len(signature & previous)/len(signature | previous)
            if similarity>=threshold:
                union(index,other);matches.append([other,index,similarity])
        for item in signature: inverted[item].append(index)
    groups=defaultdict(list)
    for index in range(len(rows)):groups[root(index)].append(index)
    result={};members={}
    for indices in groups.values():
        # Representative is independent of row order; includes transitive closure.
        identity=min(canonical(rows[index]['prompt']) for index in indices)
        key=hashlib.sha256(identity.encode()).hexdigest()
        bucket=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16],16)%100
        split='train' if bucket<80 else 'validation' if bucket<90 else 'test'
        for index in indices:result[index]=split;members[index]=key
    leaks=[pair for pair in matches if result[pair[0]]!=result[pair[1]]]
    payload=[dict(row,split=result[index],group=members[index]) for index,row in enumerate(rows)]
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload,{'method':'exhaustive_shared_3gram_jaccard_components_v1','threshold':threshold,
        'rows':len(rows),'groups':len(groups),'near_duplicate_pairs':len(matches),
        'cross_split_pairs':len(leaks),'seed':seed,'sha256':digest,
        'counts':{split:sum(item['split']==split for item in payload) for split in ('train','validation','test')}}
