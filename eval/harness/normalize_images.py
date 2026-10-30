#!/usr/bin/env python
"""Normalize every counting-bench image to a JPEG the vision APIs will accept.

The delivery mixes HEIC/WebP/GIF/extensionless files. Convert to RGB JPEG
with a maximum long side of 2048 pixels and quality 92.

in:  ~/counting-bench/tasks/Task-<id>/{image,image_secondary}/*
out: <outdir>/<Task-id>__{primary,secondary}.jpg  + manifest.csv
"""
import argparse, glob, os, csv
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
try:
    from pillow_heif import register_heif_opener; register_heif_opener()
except Exception:
    pass

MAXSIDE, QUALITY = 2048, 92

def one(args):
    """(task_dir, subdir, role, outdir) -> row dict | None"""
    tdir, sub, role, outdir = args
    files = sorted(glob.glob(os.path.join(tdir, sub, '*')))
    if not files:
        return None
    src = files[0]
    tid = os.path.basename(tdir)
    dst = os.path.join(outdir, f'{tid}__{role}.jpg')
    row = dict(task_dir=tid, role=role, src=src, dst=dst, src_bytes=os.path.getsize(src),
               w=0, h=0, out_w=0, out_h=0, resized=0, converted=0, status='ok', err='')
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        row['status'] = 'cached'
        with Image.open(dst) as im: row['out_w'], row['out_h'] = im.size
        return row
    try:
        # Batch 5 introduced SVG secondary images, which PIL cannot rasterize. Shell out to
        # rsvg-convert (librsvg2-bin) and continue through the normal path.
        if os.path.splitext(src)[1].lower() == '.svg':
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                png = tmp.name
            subprocess.run(['rsvg-convert', '-w', str(MAXSIDE), src, '-o', png],
                           check=True, capture_output=True, timeout=60)
            src_open, row['converted'] = png, 1
        else:
            src_open = src
        with Image.open(src_open) as im:
            im.load()
            row['w'], row['h'] = im.size
            if im.mode not in ('RGB', 'L'):
                im = im.convert('RGB'); row['converted'] = 1
            elif im.mode == 'L':
                im = im.convert('RGB'); row['converted'] = 1
            if max(im.size) > MAXSIDE:
                s = MAXSIDE / max(im.size)
                im = im.resize((max(1, round(im.width*s)), max(1, round(im.height*s))), Image.LANCZOS)
                row['resized'] = 1
            row['out_w'], row['out_h'] = im.size
            im.save(dst, 'JPEG', quality=QUALITY, optimize=True)
        if src_open != src:
            os.unlink(src_open)
    except Exception as e:
        row['status'] = 'error'; row['err'] = f'{type(e).__name__}: {e}'
    return row

def main():
    ap = argparse.ArgumentParser()
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--repo', default=_repo)
    ap.add_argument('--outdir', default=os.path.join(_repo, 'images'))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jobs = []
    for tdir in sorted(glob.glob(os.path.join(a.repo, 'tasks', 'Task-*'))):
        jobs.append((tdir, 'image', 'primary', a.outdir))
        jobs.append((tdir, 'image_secondary', 'secondary', a.outdir))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=8), 1):
            if r: rows.append(r)
            if i % 400 == 0: print(f'{i}/{len(jobs)}', flush=True)
    mf = os.path.join(a.outdir, 'manifest.csv')
    if not rows:
        raise SystemExit('No source images found; check --repo')
    with open(mf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = sum(r['status'] in ('ok','cached') for r in rows)
    print(f'wrote {mf}: {len(rows)} images, {ok} usable, '
          f'{sum(r["resized"] for r in rows)} downscaled, '
          f'{sum(r["converted"] for r in rows)} colour-converted, '
          f'{sum(r["status"]=="error" for r in rows)} errors')
    for r in rows:
        if r['status'] == 'error': print('  ERR', r['src'], r['err'])

if __name__ == '__main__':
    main()
