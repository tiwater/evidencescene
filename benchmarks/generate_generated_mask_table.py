"""Render pinned shared-generated-mask score summaries for the short paper."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = Path(os.environ.get('EVIDENCESCENE_RECEIPTS_DIR', ROOT / 'benchmarks' / 'receipts'))
CASES = (
    ('Cracker', 'ycbineoat-generated-mask-recall-20261028/score.json', '8089f1e636f495fe569b93433cc22ea4be80f9a10c187bffd50e68662a747675', 'cracker'),
    ('Mustard', 'ycbineoat-mustard-generated-mask-20261028/score.json', '39e8e9f699c35e2ad0ecdf91c8fb8255c56e8666f9c9a6c7a20d287dd73edcb0', 'pair'),
    ('Bleach', 'ycbineoat-bleach-generated-mask-20261028/score.json', '514942fc9b6e1cc71ec0ba057136e967d0963e843e754719c49daea15c42b041', 'pair'),
    ('Sugar', 'ycbineoat-two-object-extension-20261028/sugar-generated-score.json', '6d3864b96f34e295c7d776a6e2ceff1233f10b3d32c96722ed4497196a149b5f', 'extension'),
    ('Tomato', 'ycbineoat-two-object-extension-20261028/tomato-generated-score.json', '27cb8164c6f2722e8c33b4d0a4771637b95408504f8f799bac8567275be45f31', 'extension'),
)
print(r'\begin{tabular}{@{}lrrrrr@{}}')
print(r'\toprule')
print(r'Sequence & $n$ & P2P 10 & BT 10 & P2P 20 & BT 20 \\')
print(r'\midrule')
for label, path, digest, kind in CASES:
    raw = (P / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'source hash mismatch: {path}')
    data = json.loads(raw)
    if kind == 'cracker':
        arms = [data['point2pose'], data['bundletrack']]
        key = lambda a: (a['denominator_excluding_initial'], a['unavailable_count'], a['adds_le_10mm_count'], a['adds_le_20mm_count'])
    else:
        arms = [data['point'], data['bundle']] if kind == 'pair' else [data['point2pose'], data['bundletrack']]
        key = lambda a: (a['denominator'], a['unavailable'], a['within_10mm'], a['within_20mm'])
    point, bundle = [key(a) for a in arms]
    for n, unavailable, passing10, passing20 in (point, bundle):
        if (any(type(x) is not int for x in (n, unavailable, passing10, passing20))
                or n <= 0 or not 0 <= unavailable <= n
                or not 0 <= passing10 <= passing20 <= n - unavailable):
            raise ValueError(f'invalid outcome counts: {label}')
    if point[0] != bundle[0]:
        raise ValueError(f'denominator mismatch: {label}')
    print(f'{label} & {point[0]} & {point[2]} & {bundle[2]} & {point[3]} & {bundle[3]} ' + r'\\')
print(r'\bottomrule')
print(r'\end{tabular}')
