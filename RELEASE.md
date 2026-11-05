# Dataset release status

**Draft: metadata complete, images not published.** This is not a runnable full
dataset release yet.

The candidate is pinned to source revision
`29932f05513a1b53e192ab9b18e538a442835a88` and Attribution v5:

- 1,635 tasks with neutral, alignment-cued and multi-hop prompts and answers.
- 1,635 primary and 823 secondary images expected; exact filenames, byte counts
  and Git blob identifiers are in `dataset/media_manifest.json`.
- 25 withdrawn tasks excluded from all conditions; 13 secondary-only blurred
  images retained with exact reviewed bytes pinned.

## Before publishing images

1. Reconcile v5's **74 primary and 87 secondary FALSE redistribution records**.
   These include explicit copyrighted/not-releasable records as well as AI-origin
   flags. Obtain corrected item-level permission decisions, not a blanket override.
2. Confirm the annotation licence and retain each image's individual terms,
   credits and modification notices. Code remains MIT; it does not license data.
3. Export only the manifest's image files into `dataset/tasks/`. Do not copy source
   Git history, original delivery JSON, old attribution versions, or consent logs.
4. Run the full checker, normalize every image without errors, and test a clean
   download through the evaluator for all three conditions. Current tests cover
   metadata and a synthetic mock-API run, not the unavailable release images.

```bash
# Metadata integrity only; prints permission blockers but does not certify release.
python eval/harness/check_release.py --metadata-only
# Fails until image files and recorded permissions are complete.
python eval/harness/check_release.py
```

Do not use the paper's 1,660-item scores as results on this edited candidate.
Multi-hop answer-quality limitations remain documented in [DATASET.md](DATASET.md#limitations).
