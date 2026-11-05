"""Check candidate membership, image bytes, and recorded release permissions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition):
    if not condition:
        raise ValueError('Inconsistent release metadata')


def check(root, metadata_only=False):
    index = rows(root / 'index.csv')
    credits = rows(root / 'ATTRIBUTION.csv')
    manifest = json.loads((root / 'media_manifest.json').read_text())
    exclusions = json.loads((root / 'exclusions.json').read_text())
    names = [row['task_name'] for row in index]
    require(len(names) == len(set(names)) == manifest['tasks'])
    require({row['item_id'] for row in credits} == set(names))
    require(len(credits) == len(names))
    ids = {name.replace('Task ', 'Task-', 1) for name in names}
    require(not ids.intersection(exclusions['excluded_tasks']))
    media = manifest['files']
    paths = [entry['path'] for entry in media]
    require(len(paths) == len(set(paths)))
    for path in paths:
        parts = PurePosixPath(path).parts
        require(len(parts) == 4 and parts[0] == 'tasks' and parts[1] in ids)
        require(parts[2] in ('image', 'image_secondary') and '..' not in parts)
    primary = [path.split('/')[1] for path in paths if '/image/' in path]
    secondary = [path.split('/')[1] for path in paths if '/image_secondary/' in path]
    require(set(primary) == ids and len(primary) == len(ids))
    require(len(secondary) == len(set(secondary)) == manifest['secondary_images'])
    require(len(names) == exclusions['candidate_items'])
    require(len(secondary) == exclusions['candidate_secondary_images'])
    blobs = {entry['path']: entry['git_blob'] for entry in media}
    for path, blob in exclusions['retained_secondary_blurs'].items():
        require(blobs[path] == blob)
    for row in index:
        require(all(row[field].strip() for field in
                    ('neutral_prompt', 'cued_prompt', 'contextual_prompt')))
        int(row['true_count'])
        int(row['contextual_count'])
        require(json.loads(row['signature_count']))

    issues = []
    for row in credits:
        task = row['item_id'].replace('Task ', 'Task-', 1)
        for prefix in ('', 'second_') if task in secondary else ('',):
            if row[prefix + 'redistribution_allowed'].strip().upper() != 'TRUE':
                issues.append(f'{task}: {prefix or "primary_"}redistribution not confirmed')
            for field in ('original_license', 'attribution_text'):
                if not row[prefix + field].strip():
                    issues.append(f'{task}: missing {prefix}{field}')
    if not metadata_only:
        expected = set(paths)
        actual = {p.relative_to(root).as_posix() for p in (root / 'tasks').rglob('*')
                  if p.is_file() or p.is_symlink()}
        issues.extend(f'unexpected file: {path}' for path in sorted(actual - expected))
        for entry in media:
            path = root / entry['path']
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                issues.append(f'missing or symlinked image: {entry["path"]}')
                continue
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != entry['git_blob'] or len(data) != entry['bytes']:
                issues.append(f'changed image: {entry["path"]}')
    return issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2] / 'dataset')
    parser.add_argument('--metadata-only', action='store_true', help='Check metadata, not media readiness')
    args = parser.parse_args()
    problems = check(args.repo, args.metadata_only)
    print(f'{"METADATA ONLY" if args.metadata_only else "FULL RELEASE"}: {len(problems)} unresolved checks')
    print('\n'.join(problems[:20]))
    raise SystemExit(bool(problems) and not args.metadata_only)
