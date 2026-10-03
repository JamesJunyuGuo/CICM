import json,re,sys
from collections import defaultdict
def classify(r):
    t=(r.get("response") or "")
    if r.get("error") or not t.strip(): return "refusal"
    tl=t.lower()
    cur=r["current_value"].lower()
    stale=[s.lower() for s in r["stale_values"]]; oth=[o.lower() for o in r["other_values"]]
    # longest-first matching to avoid substring collisions
    hits=[v for v in sorted(set([cur]+stale+oth), key=len, reverse=True) if re.search(r'\b'+re.escape(v)+r'\b', tl)]
    if not hits: return "none"
    # take the value the model asserts: if only one distinct known value present, use it; else last-occurring
    if cur in hits and all(h==cur or tl.rfind(cur)>tl.rfind(h) for h in hits): return "correct"
    # pick the known value that appears last in the text
    pick=max(hits, key=lambda v: tl.rfind(v))
    return "correct" if pick==cur else "stale" if pick in stale else "cross" if pick in oth else "none"
cell=defaultdict(lambda:defaultdict(int))
for l in open(sys.argv[1]):
    r=json.loads(l); cell[(r["k"],r["control"])][classify(r)]+=1
print(" k cond  acc  stale cross none ref")
for (k,c) in sorted(cell):
    d=cell[(k,c)];n=sum(d.values());ans=n-d['refusal'];acc=d['correct']/ans if ans else 0
    print(f"{k:>2} {'ctrl' if c else 'ovw':<4} {acc:.2f}  {d['stale']:>3}  {d['cross']:>3}  {d['none']:>3} {d['refusal']:>3}")
