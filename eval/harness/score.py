#!/usr/bin/env python
"""Score a counting-bench run. Re-parses answers from the stored raw text, so a parser
fix never requires re-calling the API.

Metrics, per model x condition:
  acc          exact-match accuracy (the benchmark's headline: "Return an integer")
  off_by_1     |pred-gt| == 1  (a near-miss on a dense count is not the same failure as a decoy bite)
  decoy_hit    pred equals a planted decoy count and is wrong -- what the bench is built to measure
  mape         median absolute % error, magnitude-robust
  signed_bias  median (pred-gt)/gt -- are models under- or over-counting?
  no_answer    unparseable/empty/errored
"""
import argparse, json, os, collections, math
import pandas as pd, numpy as np
from run_counting import parse_int

def open_responses(run_dir):
    """Open a run's responses, transparently handling the gzipped form.

    Committed runs ship `responses.jsonl.gz` to keep the repo small; a live run writes
    `responses.jsonl`. Readers that only knew about the plain form made the released artifacts
    un-analysable, which defeats the point of committing them.
    """
    import gzip, os
    plain = os.path.join(run_dir, 'responses.jsonl')
    gz = plain + '.gz'
    if os.path.exists(plain):
        return open(plain, 'r')
    if os.path.exists(gz):
        return gzip.open(gz, 'rt')
    raise FileNotFoundError(f'no responses.jsonl or responses.jsonl.gz in {run_dir}')

def load(runs):
    rows = []
    for run in runs:
        seen = {}
        with open_responses(run) as responses:
            for line in responses:
                try: d = json.loads(line)
                except Exception: continue
                seen[(d['model'], d['task_dir'], d['condition'])] = d   # later line wins (resume)
        for d in seen.values():
            if d.get('applicable') is False:
                continue          # structurally inapplicable: excluded from the denominator
            p, mode = parse_int(d.get('raw') or '')
            if d.get('error'): p, mode = None, 'api_error'
            gt = d['gt']
            dec = [c for c in (d.get('decoy_counts') or []) if c is not None]
            rows.append(dict(run=os.path.basename(str(run).rstrip('/')), model=d['model'], task=d['task_dir'],
                condition=d['condition'], gt=gt, pred=p, parse_mode=mode,
                correct=(p == gt), off_by_1=(p is not None and abs(p-gt) == 1),
                decoy_hit=(p is not None and p != gt and p in dec),
                has_decoy=len(dec) > 0, n_decoys=len(dec),
                decoy_cats=';'.join(sorted({c for c in (d.get('decoy_categories') or []) if c})),
                abs_err=(abs(p-gt) if p is not None else np.nan),
                pct_err=((abs(p-gt)/gt*100) if (p is not None and gt) else np.nan),
                signed_pct=(((p-gt)/gt*100) if (p is not None and gt) else np.nan),
                no_answer=(p is None), n_images=d.get('n_images'),
                scene_domain=d.get('scene_domain'), image_origin=d.get('image_origin'),
                finish_reason=d.get('finish_reason'), latency_s=d.get('latency_s'),
                effort=d.get('effort'),
                prompt_tokens=(d.get('usage') or {}).get('prompt_tokens'),
                completion_tokens=(d.get('usage') or {}).get('completion_tokens')))
    return pd.DataFrame(rows)

def wilson(k, n, z=1.96):
    if not n: return (float('nan'),)*2
    p = k/n; d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h)*100, min(1, c+h)*100)

def summarize(df):
    g = []
    for (m, c), s in df.groupby(['model', 'condition']):
        n = len(s); k = int(s.correct.sum())
        lo, hi = wilson(k, n)
        g.append(dict(model=m, condition=c, n=n, acc=100*k/n, ci_lo=lo, ci_hi=hi,
                      off_by_1=100*s.off_by_1.mean(), decoy_hit=100*s.decoy_hit.mean(),
                      mape=s.pct_err.median(), signed_bias=s.signed_pct.median(),
                      no_answer=100*s.no_answer.mean(),
                      med_compl_tok=s.completion_tokens.median(),
                      med_lat=s.latency_s.median()))
    return pd.DataFrame(g).sort_values(['condition', 'acc'], ascending=[True, False])

def mcnemar(df, m1, m2, cond):
    a = df[(df.model == m1) & (df.condition == cond)].set_index('task').correct
    b = df[(df.model == m2) & (df.condition == cond)].set_index('task').correct
    j = a.to_frame('a').join(b.to_frame('b'), how='inner').dropna()
    n01 = int(((~j.a) & j.b).sum()); n10 = int((j.a & (~j.b)).sum())
    if n01 + n10 == 0: return len(j), n10, n01, float('nan')
    from math import comb
    n = n01 + n10; k = min(n01, n10)
    p = min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n)
    return len(j), n10, n01, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    df = load(a.runs)
    pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 200)
    print(f'== records: {len(df)}  models: {sorted(df.model.unique())}  tasks: {df.task.nunique()}\n')
    s = summarize(df)
    print('== headline (acc = exact match, 95% Wilson CI) ==')
    print(s.round(1).to_string(index=False))
    print('\n== parse modes ==');  print(df.parse_mode.value_counts().to_string())
    print('\n== finish reasons =='); print(df.finish_reason.value_counts(dropna=False).to_string())

    print('\n== accuracy by count magnitude (neutral+cued) ==')
    d2 = df[df.condition.isin(['neutral', 'cued'])].copy()
    d2['bucket'] = pd.cut(d2['gt'], [0, 5, 10, 20, 50, 100, 10**7],
                          labels=['1-5', '6-10', '11-20', '21-50', '51-100', '100+'])
    print((d2.groupby(['model', 'bucket'], observed=True)
             .agg(n=('correct', 'size'), acc=('correct', lambda x: 100*x.mean()),
                  decoy=('decoy_hit', lambda x: 100*x.mean())).round(1).to_string()))

    print('\n== decoy-hit rate by decoy category (single-decoy tasks only) ==')
    sd = df[(df.n_decoys == 1) & (df.condition.isin(['neutral', 'cued']))]
    print((sd.groupby(['decoy_cats', 'model'])
             .agg(n=('correct', 'size'), acc=('correct', lambda x: 100*x.mean()),
                  decoy_hit=('decoy_hit', lambda x: 100*x.mean())).round(1).to_string()))

    print('\n== cueing effect (paired, same tasks: cued - neutral) ==')
    for m in sorted(df.model.unique()):
        nu = df[(df.model == m) & (df.condition == 'neutral')].set_index('task').correct
        cu = df[(df.model == m) & (df.condition == 'cued')].set_index('task').correct
        j = nu.to_frame('n').join(cu.to_frame('c'), how='inner').dropna()
        if len(j) == 0: continue
        n01 = int(((~j.n) & j.c).sum()); n10 = int((j.n & (~j.c)).sum())
        print(f'  {m:20s} n={len(j)} neutral={100*j.n.mean():.1f}% cued={100*j.c.mean():.1f}% '
              f'delta={100*(j.c.mean()-j.n.mean()):+.1f}pp  (fixed {n01}, broke {n10})')

    print('\n== by scene_domain (neutral) ==')
    print((df[df.condition == 'neutral'].groupby(['scene_domain', 'model'])
             .agg(n=('correct', 'size'), acc=('correct', lambda x: 100*x.mean())).round(1).to_string()))
    print('\n== by image_origin (neutral) ==')
    print((df[df.condition == 'neutral'].groupby(['image_origin', 'model'])
             .agg(n=('correct', 'size'), acc=('correct', lambda x: 100*x.mean())).round(1).to_string()))

    ms = sorted(df.model.unique())
    if len(ms) > 1:
        print('\n== McNemar, paired by task ==')
        for c in ['neutral', 'cued', 'contextual']:
            for i in range(len(ms)):
                for j in range(i+1, len(ms)):
                    n, w, l, p = mcnemar(df, ms[i], ms[j], c)
                    print(f'  {c:11s} {ms[i]} vs {ms[j]}: n={n} '
                          f'{ms[i]}-only={w} {ms[j]}-only={l} p={p:.2g}')
    tot = df.completion_tokens.sum(skipna=True), df.prompt_tokens.sum(skipna=True)
    print(f'\n== tokens: prompt {tot[1]:,.0f}  completion {tot[0]:,.0f}')
    if a.out:
        df.to_csv(a.out, index=False); print(f'per-call rows -> {a.out}')
        s.to_csv(a.out.replace('.csv', '_summary.csv'), index=False)

if __name__ == '__main__':
    main()
