"""Build the manuscript's primary table from committed local score receipts."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=Path(os.environ.get('EVIDENCESCENE_RECEIPTS_DIR', ROOT/'benchmarks'/'receipts'))
def read(relative, digest):
 data=(P/relative).read_bytes()
 if hashlib.sha256(data).hexdigest()!=digest:raise ValueError(f'score changed: {relative}')
 return json.loads(data)
base=read('ycbineoat-public-recall-20261028/score.json','1ec8c55d785851bb6e9d77b37d584d78cc02f42867f478e0c7ec50dacdca9a57')['cases']
bleach=read('ycbineoat-bleach0-development-20261028/results/recall.json','e1e67c9b889863282e41a9a16788d298290b3c6845375f5344a55e2528abe814')['result']
sugar=read('ycbineoat-two-object-extension-20261028/sugar-score.json','ff35f1d6b50d0a047ab2464822e9b90486147fc43f0c6029c15927a8cbc5d7ce')
tomato=read('ycbineoat-two-object-extension-20261028/tomato-score.json','6f72aabe53bdea695ef6773e394d8e2a585cd73b00ea83f69c5889a97fc8bb89')
hard=read('ycbineoat-bleach-hard-development-20261029/score.json','0fb9abc82ce5cf2920906d389ff1782191e948574e1ff6c1cbb71a37d154293b')
rows=[('Cracker',base[0]),('Mustard',base[1]),('Bleach',bleach),('Sugar',sugar),('Tomato',tomato),('Bleach-hard*',hard)]
print(r'\begin{tabular}{@{}lrrrrr@{}}')
print(r'\toprule')
print(r'Sequence & $n$ & P2P 10 & BT 10 & P2P AUC & BT AUC \\')
print(r'\midrule')
for name,r in rows:
 a,b=r['point2pose'],r['bundletrack']; n=a['denominator_excluding_initial']
 if n!=b['denominator_excluding_initial']:raise ValueError('denominator mismatch')
 print(f"{name} & {n} & {a['adds_le_10mm_count']} & {b['adds_le_10mm_count']} & {a['recall_auc_0_to_100mm']:.3f} & {b['recall_auc_0_to_100mm']:.3f} " + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')
