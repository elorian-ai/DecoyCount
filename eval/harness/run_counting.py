#!/usr/bin/env python
"""Standalone DecoyCount API evaluator.

Each task has 1-2 images and three prompt conditions graded against an
integer ground truth:
  neutral    -> true_count        (plain "how many X?")
  cued       -> true_count        (same + an anti-decoy hint)
  contextual -> contextual_count  (arithmetic over the count, often across both images)

Every task also ships `decoys` = [{sig_note, sig_count, sig_category}]: the number the
image is engineered to bait the model into saying. We score exact-match accuracy AND
decoy-hit rate, because the second is what the benchmark was built to measure.

Output: results/<run>/responses.jsonl, one line per (model, task, condition). Resumable --
re-running skips keys already present, so a killed run continues where it stopped.
"""
import argparse, json, os, re, sys, time, threading, random, platform, subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from providers import (PROVIDERS, ProviderError, parse_model_spec, make_client,
                       build_request, describe)

HARNESS_VERSION = '2.0.0'

def _git_sha():
    """Commit of the harness itself, so a result can be traced to the code that made it."""
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None

def _pkg_version(name):
    try:
        from importlib.metadata import version; return version(name)
    except Exception:
        return None

# The three shipped conditions, plus two prompt controls we construct (see build_cue_controls).
# They exist to decompose the neutral->cued gain: is it targeted re-inspection, or generic compliance?
#   generic : neutral + a fixed "try harder" instruction that names no trap
#   falsecue: neutral + a REAL cue borrowed from another item whose decoy families are disjoint,
#             so the instruction is specific and confident but points at the wrong thing
CONDITIONS = {'neutral': 'true_count', 'cued': 'true_count', 'contextual': 'contextual_count',
              'generic': 'true_count', 'falsecue': 'true_count',
              # paraphrases of the item's OWN cue: same content, different wording. If the cueing gain
              # is a wording artefact these diverge; if it is about content they track `cued`.
              'para1': 'true_count', 'para2': 'true_count', 'para3': 'true_count',
              # contextual with the two images supplied in the opposite order (position control).
              'ctxswap': 'contextual_count'}
PARAPHRASE = [
    lambda c: f'Before answering, note: {c[0].lower()+c[1:]}.',
    lambda c: f'It is important that you {c[0].lower()+c[1:]}.',
    lambda c: f'Have you made sure to {c[0].lower()+c[1:]}?',
]
GENERIC_CUE = 'Look carefully and count precisely.'
RETURN_INT = 'Return an integer.'

def _cue_suffix(neutral, cued):
    """The instruction `cued` adds to `neutral`, or None if it is not a clean extension."""
    n, c = str(neutral).strip(), str(cued).strip()
    nb = n[:-len(RETURN_INT)].strip() if n.endswith(RETURN_INT) else n
    cb = c[:-len(RETURN_INT)].strip() if c.endswith(RETURN_INT) else c
    if not cb.lower().startswith(nb.lower()[:max(8, len(nb)//2)]):
        return None
    i = 0
    while i < min(len(nb), len(cb)) and nb[i].lower() == cb[i].lower():
        i += 1
    suf = cb[i:].strip(' .')
    return suf if 3 < len(suf) < 400 else None

def build_cue_controls(tasks, seed=1234):
    """Attach `generic_prompt` and `falsecue_prompt` to each task, in place.

    A false cue is donated by another task whose decoy families do NOT intersect this one's, so the
    borrowed instruction is genuinely inapplicable. Deterministic given the seed.
    """
    rng = random.Random(seed)
    STOP = {'the','a','an','how','many','are','there','is','of','in','on','and','or','to','be','it',
            'this','that','these','those','return','integer','count','counting','each','every','only',
            'do','not','their','its','all','any','with','for','from','at','by','as','if','they','them'}
    def _object_agnostic(neutral, suffix):
        """True if the cue never names the donor's own subject, so it can be transplanted without
        telling the model to count a different thing. Without this filter a false cue like
        'Count complete pieces of donuts' on an image of albums confounds trap-mismatch with
        object-confusion, and we would not know which one moved the number."""
        import re as _re
        def words(s):
            out = set()
            for w in _re.findall(r'[a-z]+', str(s).lower()):
                if w in STOP or len(w) <= 2: continue
                out.add(w)
                out.add(w[:-1] if w.endswith('s') else w + 's')   # crude singular/plural tie
            return out
        return not (words(neutral) & words(suffix))

    donors = []
    for t in tasks:
        suf = _cue_suffix(t['neutral_prompt'], t['cued_prompt'])
        if suf and _object_agnostic(t['neutral_prompt'], suf):
            donors.append((t['task_dir'], suf,
                           frozenset(d.get('sig_category') for d in t['decoys'])))
    n_false = 0
    for t in tasks:
        base = str(t['neutral_prompt']).strip()
        stem = base[:-len(RETURN_INT)].strip() if base.endswith(RETURN_INT) else base
        t['generic_prompt'] = f'{stem} {GENERIC_CUE} {RETURN_INT}'
        own = _cue_suffix(t['neutral_prompt'], t['cued_prompt'])
        for i, f in enumerate(PARAPHRASE, 1):
            t[f'para{i}_prompt'] = f'{stem} {f(own)} {RETURN_INT}' if own else None
        t['ctxswap_prompt'] = t['contextual_prompt']
        mine = frozenset(d.get('sig_category') for d in t['decoys'])
        pool = [d for d in donors if d[0] != t['task_dir'] and not (d[2] & mine)]
        if pool:
            _, suf, _ = pool[rng.randrange(len(pool))]
            t['falsecue_prompt'] = f'{stem} {suf}. {RETURN_INT}'
            n_false += 1
        else:
            t['falsecue_prompt'] = None
    return n_false
_print_lock = threading.Lock()

# ---------------------------------------------------------------- answer parsing
_INT = re.compile(r'-?\d[\d,]*')
_BOLD_INT = re.compile(r'\*\*\s*(-?\d[\d,]*)\s*\*\*')

def parse_int(text):
    """model text -> (int|None, parse_mode).

    Every prompt ends with "Return an integer.", but models still narrate. The answer is
    the LAST number stated, not the most emphatic one: an early "Total on the rack: 11"
    is a working step, the trailing "5" is the answer. An earlier version preferred a
    line labelled total/answer/count and mis-scored those (read 5 as 3, 34 as 56), so
    position now wins and the label is only a tie-breaker inside the chosen line.
    """
    if not text: return None, 'empty'
    t = text.strip().strip('`').strip()
    whole = t.replace('**', '').replace(',', '').strip()
    if re.fullmatch(r'-?\d+', whole):
        return int(whole), 'exact'
    t = t.replace('$', '')                      # models wrap arithmetic in LaTeX
    for line in reversed([l for l in t.splitlines() if l.strip()]):
        bold = _BOLD_INT.findall(line)
        if bold:                                # a bolded number in the final line is the answer
            return int(bold[-1].replace(',', '')), 'bold_last_line'
        hits = _INT.findall(line)
        if hits:
            return int(hits[-1].replace(',', '')), 'last_line'
    return None, 'no_int'

# ---------------------------------------------------------------- request
_B64_CACHE: 'OrderedDict[str, str]' = OrderedDict()
_B64_CACHE_MAX = 256
_B64_LOCK = threading.Lock()

def b64_image(path):
    """Base64 of one image, bounded LRU, safe under the default 128 worker threads.

    The lock is not optional. An earlier version read the cache and then called move_to_end
    outside any lock; between those two steps another thread's miss could evict that exact key
    and move_to_end raised KeyError, which propagates out of ThreadPoolExecutor.map and aborts
    the sweep. Bounding the cache to fix memory introduced a crash; the lock is what makes the
    bound safe.
    """
    import base64
    with _B64_LOCK:
        hit = _B64_CACHE.get(path)
        if hit is not None:
            _B64_CACHE.move_to_end(path)
            return hit
    with open(path, 'rb') as f:                       # read outside the lock
        enc = base64.b64encode(f.read()).decode()
    with _B64_LOCK:
        _B64_CACHE[path] = enc
        _B64_CACHE.move_to_end(path)
        while len(_B64_CACHE) > _B64_CACHE_MAX:
            _B64_CACHE.popitem(last=False)
    return enc

def build_messages(prompt, image_paths):
    content = [{'type': 'text', 'text': prompt}]
    for p in image_paths:
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64_image(p)}'}})
    return [{'role': 'user', 'content': content}]

# Providers disagree about which optional parameters exist, and behind a gateway the harness
# cannot know what the underlying model accepts. Rather than make the caller memorise that,
# drop the offending parameter and retry once. What was dropped is recorded on the row.
DROPPABLE = ('reasoning_effort',)

def _unsupported_param(msg: str):
    """-> the parameter name the provider rejected, if that is what went wrong."""
    low = msg.lower()
    if 'unsupported' not in low and 'not support' not in low and 'unknown parameter' not in low:
        return None
    for name in DROPPABLE:
        if name in low:
            return name
    return None

def call_model(client, kw, retries):
    """Issue one request. `kw` is already shaped for the provider by build_request()."""
    kw = dict(kw)
    backoff = 4.0
    last = None
    dropped = []
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = client.chat.completions.create(**kw)
            ch = r.choices[0]
            return dict(text=(ch.message.content or ''), finish_reason=ch.finish_reason,
                        latency_s=round(time.time() - t0, 2),
                        usage=(r.usage.model_dump() if r.usage else None),
                        attempts=attempt + 1, error=None,
                        dropped_params=(dropped or None))
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            bad = _unsupported_param(last)
            if bad and bad in kw:
                del kw[bad]; dropped.append(bad)
                continue                      # immediate retry, this is not a transient failure
            if attempt < retries:
                time.sleep(backoff + random.random() * 2); backoff = min(backoff * 2, 90)
    return dict(text='', finish_reason=None, latency_s=None, usage=None,
                attempts=retries + 1, error=last, dropped_params=(dropped or None))

# ---------------------------------------------------------------- data
def _to_int(v, default=None):
    """Tolerate a blank or NaN cell. An unguarded int() on one empty contextual_count aborted the
    whole dataset load before a single call was made -- including for --conditions neutral, which
    never reads that field."""
    try:
        if v is None: return default
        s = str(v).strip().replace(',', '')
        if s == '' or s.lower() in {'nan', 'none'}: return default
        return int(float(s))
    except (TypeError, ValueError):
        return default

def load_tasks(repo, imgdir):
    df = pd.read_csv(os.path.join(repo, 'index.csv'))
    if df.empty or df.task_name.isna().any() or df.task_name.duplicated().any():
        raise ValueError('index.csv must contain non-empty, unique task names')
    df['task_dir'] = 'Task-' + df.task_name.str.replace('Task ', '', regex=False)
    tasks = []
    for _, r in df.iterrows():
        prim = os.path.join(imgdir, f'{r.task_dir}__primary.jpg')
        sec = os.path.join(imgdir, f'{r.task_dir}__secondary.jpg')
        if not os.path.exists(prim):
            raise FileNotFoundError(f'missing normalized primary image: {prim}')
        try:
            decoys = json.loads(r.signature_count) if isinstance(r.signature_count, str) else []
        except Exception:
            decoys = []
        tasks.append(dict(task_dir=r.task_dir, studio_id=r.studio_id,
                          primary=prim, secondary=sec if os.path.exists(sec) else None,
                          neutral_prompt=r.neutral_prompt, cued_prompt=r.cued_prompt,
                          contextual_prompt=r.contextual_prompt,
                          true_count=_to_int(r.true_count), contextual_count=_to_int(r.contextual_count),
                          decoys=decoys, scene_domain=r.scene_domain, image_origin=r.image_origin))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    # Defaults derive from this file's location, so a fresh clone works with no flags and no
    # path on any developer's machine is ever baked in. harness/ -> eval/ -> repo root.
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--repo', default=_repo,
                    help='dataset root containing index.csv and tasks/ (default: this clone)')
    ap.add_argument('--imgdir', default=os.path.join(_repo, 'images'),
                    help='normalized images from normalize_images.py (default: <clone>/images)')
    ap.add_argument('--out', help='run dir under results/ (required unless --list-providers)')
    ap.add_argument('--models', nargs='+',
                    help='alias=provider:model_id, e.g. qwen3.8-flash=dashscope:qwen3.8-flash. '
                         'Run --list-providers to see providers and their key env vars.')
    ap.add_argument('--list-providers', action='store_true',
                    help='print the provider table and exit')
    SHIPPED = ['neutral', 'cued', 'contextual']
    ap.add_argument('--conditions', nargs='+', default=SHIPPED, choices=list(CONDITIONS),
                    help='default: the three conditions shipped with each item. The others '
                         '(generic, falsecue, para1-3, ctxswap) are controls we construct.')
    ap.add_argument('--limit', type=int, default=0, help='0 = all tasks')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--no-media', action='store_true',
                    help='text-only control: send the prompt WITHOUT any image. Bounds how much of the '
                         'benchmark is solvable from linguistic priors alone.')
    ap.add_argument('--tag', default='', help='free-text label recorded on every row (e.g. repeat index)')
    ap.add_argument('--max-tokens', type=int, default=16384)
    # Gemini is latency-bound: measured 474-520 calls/min at 128 workers vs 54/min at 12, with
    # identical error counts. Low concurrency is never the right choice here, so 128 is the default
    # rather than something a caller has to remember to pass.
    ap.add_argument('--workers', type=int, default=128)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--retries', type=int, default=4)

    a = ap.parse_args()
    if a.list_providers:
        print(describe()); return
    if not a.out or not a.models:
        ap.error('--out and --models are required (or pass --list-providers)')
    if a.workers < 1 or a.limit < 0 or a.retries < 0:
        ap.error('--workers must be positive; --limit and --retries must be nonnegative')

    os.makedirs(a.out, exist_ok=True)
    tasks = load_tasks(a.repo, a.imgdir)
    CONSTRUCTED = {'generic', 'falsecue', 'para1', 'para2', 'para3', 'ctxswap'}
    if CONSTRUCTED & set(a.conditions):
        n_false = build_cue_controls(tasks, seed=a.seed)
        print(f'cue controls built: generic for {len(tasks)}, false-cue for {n_false}')
    if a.limit:
        random.Random(a.seed).shuffle(tasks)       # deterministic subset, reproducible
        tasks = sorted(tasks[:a.limit], key=lambda t: t['task_dir'])
    # alias -> (provider, model_id); the provider is mandatory so a run is never ambiguous
    try:
        specs = [parse_model_spec(m) for m in a.models]
    except ProviderError as e:
        sys.exit(f'error: {e}')
    models = {alias: f'{prov}:{mid}' for alias, prov, mid in specs}

    outfile = os.path.join(a.out, 'responses.jsonl')
    done = set()
    if os.path.exists(outfile):
        with open(outfile) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # A row is settled if it produced an answer OR is structurally inapplicable.
                    settled = (d.get('error') is None and d.get('parsed') is not None) or \
                              (d.get('applicable') is False)
                    if settled:
                        done.add((d['model'], d['task_dir'], d['condition']))
                except Exception:
                    pass
    jobs = [(al, prov, mid, t, c) for al, prov, mid in specs for t in tasks for c in a.conditions
            if (al, t['task_dir'], c) not in done]
    n_noqt = sum(1 for _, _, _, t, c in jobs if t[CONDITIONS[c]] is None)
    if n_noqt:
        print(f'note: {n_noqt} (task, condition) pairs have no recorded answer and will be '
              f'recorded as not-applicable rather than called')
    print(f'{len(tasks)} tasks x {len(a.conditions)} conditions x {len(models)} models '
          f'= {len(tasks)*len(a.conditions)*len(models)} calls; {len(done)} already done, '
          f'{len(jobs)} to run', flush=True)

    # Provenance records the run, not the machine: no hostname, and paths are basenames only.
    # An absolute path leaks a username and a directory layout for no reproducibility benefit.
    safe = {k: v for k, v in vars(a).items()}
    for k in ('repo', 'imgdir', 'out'):
        if safe.get(k): safe[k] = os.path.basename(str(safe[k]).rstrip('/'))
    provenance = dict(safe, n_tasks=len(tasks), models=models,
                harness_version=HARNESS_VERSION,
                harness_git_sha=_git_sha(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                python=platform.python_version(),
                openai_client=_pkg_version('openai'), n_calls_planned=len(jobs))
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    for name in ('run_config.json', f'run_config_{stamp}.json'):   # latest + immutable history
        with open(os.path.join(a.out, name), 'w') as f:
            json.dump(provenance, f, indent=2)

    clients = {}
    for _, prov, _ in specs:
        if prov in clients: continue
        try:
            clients[prov] = make_client(prov)
        except ProviderError as e:
            sys.exit(f'error: {e}')
    fh = open(outfile, 'a')
    fl = threading.Lock()
    counter = {'n': 0, 'ok': 0, 'err': 0, 'skip': 0}
    t_start = time.time()

    def work(job):
        alias, prov, model_id, t, cond = job
        prompt = t[f'{cond}_prompt']
        # A missing ground truth makes the condition unscoreable. Writing gt=None instead let the
        # sweep finish and bill every call, then crashed both readers in abs(p-gt); and had it not
        # crashed, correct=(pred==None) is permanently False, planting a guaranteed miss in the
        # denominator. Treat it as inapplicable, which is what it is.
        if t[CONDITIONS[cond]] is None:
            prompt = None
        if prompt is None:
            # A constructed condition can be inapplicable to an item (no disjoint cue donor, or a
            # cued prompt that is not a clean extension of the neutral one). Record it instead of
            # returning: a silently missing row makes a run look complete when it is not. The
            # paraphrase sweep finished 4410 of 4569 this way before it was caught.
            with fl:
                fh.write(json.dumps(dict(
                    model=alias, provider=prov, model_id=model_id, task_dir=t['task_dir'],
                    studio_id=t['studio_id'], condition=cond, prompt=None, n_images=0,
                    gt=t[CONDITIONS[cond]], true_count=t['true_count'],
                    contextual_count=t['contextual_count'], parsed=None,
                    parse_mode='not_applicable', correct=False, decoy_counts=[], decoy_hit=False,
                    decoy_categories=[], scene_domain=t['scene_domain'],
                    image_origin=t['image_origin'], effort=a.effort, no_media=a.no_media,
                    tag=a.tag, raw='', finish_reason=None, latency_s=None, usage=None, attempts=0,
                    # `applicable` is a separate flag on purpose. Putting this in `error` made the
                    # readers treat it as an API failure and score the item as wrong, which biased
                    # exactly the paired paraphrase-vs-cued comparison the run exists to make.
                    applicable=False, error=None, dropped_params=None)) + '\n')
                fh.flush(); counter['n'] += 1; counter['skip'] = counter.get('skip', 0) + 1
            return
        # Only the contextual condition is written against two images; sending the
        # secondary elsewhere would change what "how many" refers to.
        if a.no_media:
            imgs = []                                  # text-only control
        else:
            imgs = [t['primary']]
            if cond in ('contextual', 'ctxswap') and t['secondary']:
                imgs.append(t['secondary'])
            if cond == 'ctxswap' and len(imgs) == 2:
                imgs.reverse()                         # position control
        client, pmeta = clients[prov]
        kw, static_dropped, eff_effort = build_request(
            pmeta, model_id, build_messages(prompt, imgs), a.effort, a.max_tokens, a.timeout)
        res = call_model(client, kw, a.retries)
        # merge the statically-dropped params with anything the retry loop had to drop
        all_dropped = static_dropped + list(res.get('dropped_params') or [])
        res['dropped_params'] = all_dropped or None
        # One rule covers both drop paths: if reasoning_effort never reached the provider, the row
        # must not claim it did. effort_curve buckets on this column, so a gateway run whose effort
        # was dropped dynamically would otherwise land in the `high` cell having sent nothing.
        if 'reasoning_effort' in all_dropped:
            eff_effort = 'none'
        parsed, mode = parse_int(res['text'])
        gt = t[CONDITIONS[cond]]
        decoy_counts = [d.get('sig_count') for d in t['decoys'] if d.get('sig_count') is not None]
        rec = dict(model=alias, provider=prov, model_id=model_id, applicable=True, task_dir=t['task_dir'], studio_id=t['studio_id'],
                   condition=cond, prompt=prompt, n_images=len(imgs),
                   gt=gt, true_count=t['true_count'], contextual_count=t['contextual_count'],
                   parsed=parsed, parse_mode=mode, correct=(parsed == gt),
                   decoy_counts=decoy_counts,
                   decoy_hit=(parsed is not None and parsed in decoy_counts and parsed != gt),
                   decoy_categories=[d.get('sig_category') for d in t['decoys']],
                   scene_domain=t['scene_domain'], image_origin=t['image_origin'],
                   effort=eff_effort, effort_requested=a.effort,
                   no_media=a.no_media, tag=a.tag, raw=res['text'][-4000:], **{k: res[k] for k in
                   ('finish_reason', 'latency_s', 'usage', 'attempts', 'error',
                    'dropped_params')})
        with fl:
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            counter['n'] += 1
            counter['err' if res['error'] else 'ok'] += 1
            if counter['n'] % 25 == 0 or counter['n'] == len(jobs):
                el = time.time() - t_start
                rate = counter['n'] / el
                print(f"  {counter['n']}/{len(jobs)}  ok={counter['ok']} err={counter['err']} "
                      f"{rate*60:.1f}/min  eta {(len(jobs)-counter['n'])/max(rate,1e-9)/60:.1f}min",
                      flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    fh.close()
    provenance.update(finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                n_ok=counter['ok'], n_error=counter['err'],
                wall_minutes=round((time.time()-t_start)/60, 2))
    for name in ('run_config.json', f'run_config_{stamp}.json'):
        with open(os.path.join(a.out, name), 'w') as f:
            json.dump(provenance, f, indent=2)
    print(f"done: {counter['ok']} ok, {counter['err']} errors, "
          f"{counter.get('skip', 0)} not-applicable in {(time.time()-t_start)/60:.1f} min")
    if counter['n'] != len(jobs):
        print(f"WARNING: wrote {counter['n']} rows for {len(jobs)} planned jobs -- investigate")

if __name__ == '__main__':
    main()
