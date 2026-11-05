# Dataset release status

The v6 preview contains **1,586 tasks, 2,108 decoys and 2,369 original images**
(1,586 primary + 783 secondary), with all three prompt conditions and answers.
It is pinned to source revision `14c697748c0ef660ee332fa0d3e685ed80df13a9`.

Every included primary and secondary image has a TRUE permission flag in
Attribution v6. AI-generated images are retained where v6 permits them. Whole
tasks are excluded if either supplied image has a FALSE flag: 31 primary failures
and 22 secondary failures overlap in four tasks, yielding **49 exclusions** from
the 1,635-task candidate. With 25 earlier withdrawals, 74 of the paper's original
1,660 tasks are excluded. All prompt conditions use the same retained task list.

## Download

Download every `decoycount-v6-part-*.zip` and `SHA256SUMS` from the
[media preview](https://github.com/elorian-ai/DecoyCount/releases/tag/v6-media-preview).
Verify the ZIPs and extract all parts into the same new directory:

```bash
shasum -a 256 -c SHA256SUMS
for part in decoycount-v6-part-*.zip; do unzip -n "$part" -d DecoyCount; done
cd DecoyCount
pip install -r requirements.txt
python eval/harness/check_release.py --media-only
python eval/harness/normalize_images.py --repo dataset --outdir images
```

Part 1 includes metadata, attribution, evaluator code and documentation. Other
parts contain additional original image files. No private Git history, consent
logs, historical delivery JSON or excluded images are included. The 13 retained
secondary-only blurred images match their pinned source bytes exactly.
Uncompressed media total approximately 3.9 GB.

## Remaining metadata limitations

Permission flags do not replace the source terms. Five primary-image records
remain incomplete in v6:

- `Task-8e9023cc`, `Task-g3r1852f`, `Task-uv3f64bc`: missing source licence.
- `Task-oa0b0094`: missing credit text (explicitly fully AI-generated).
- `Task-r3o1f791`: missing source licence and credit text.

Those source values are preserved; this preview does not certify their completeness
or grant blanket rights. Confirm the annotation licence separately. MIT covers
evaluator code only; individual image terms remain in `dataset/ATTRIBUTION.csv`.

```bash
# Membership, prompt and attribution checks; reports incomplete source fields.
python eval/harness/check_release.py --metadata-only
# Also checks images and permission metadata; incomplete fields still fail.
python eval/harness/check_release.py
```

Historical scores apply to the paper's original 1,660-item bank, not this edited
candidate. Multi-hop answer-quality limitations remain documented in
[DATASET.md](DATASET.md#limitations).
