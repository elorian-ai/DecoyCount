# Dataset format

The benchmark contains 1,660 paired counting items; 842 have a second image.
It is intended for evaluation, not training. See the [paper](https://arxiv.org/abs/2610.34855)
for collection, taxonomy and experiments.

The standalone evaluator accepts the canonical local delivery layout:

```text
data/
  index.csv
  tasks/Task-<id>/image/<primary-file>
  tasks/Task-<id>/image_secondary/<optional-secondary-file>
```

`index.csv` columns: `task_name`, `studio_id`, `neutral_prompt`, `cued_prompt`,
`contextual_prompt`, `true_count`, `contextual_count`, `signature_count`,
`scene_domain`, and `image_origin`. Source metadata may additionally include
`source_link` and `second_source_link`. `signature_count` is a JSON list of
`sig_count`, `sig_category`, and `sig_note` decoy records.

`normalize_images.py` writes `Task-<id>__primary.jpg` and optional
`Task-<id>__secondary.jpg` at a maximum long side of 2,048 pixels. Only multi-hop
receives the secondary image. The evaluator retains the legacy condition name
`contextual`; the public name is **multi-hop**.

The planned Hugging Face release may use different column names; it must be
materialized into this layout before using this evaluator. This initial code
release does not include a Hugging Face-to-local conversion tool or the media.

## Metrics

- Exact-match accuracy: predicted integer equals the annotated answer; API and
  parsing failures remain in the denominator.
- Decoy-hit rate: an incorrect prediction equals a registered decoy; interpret
  this for neutral/cued counting, not derived multi-hop answers.
- Reasoning-alignment gap: cued accuracy minus neutral accuracy over paired items,
  in percentage points. Compare the same checkpoint, dataset version and settings.

## Limitations

The internal evaluated bank flags 121 potentially stale multi-hop answers because
their underlying counts changed without revalidation of the derived answers.
This does not establish that all 121 are wrong, or certify the remaining labels.
The source index does not contain that flag. The standalone evaluator does not
automatically exclude those items: contextual scores need the release's item-level
quality metadata and must state the population used.

Alignment cues may also disambiguate a question; a positive gap does not uniquely
isolate every reasoning failure. Decoy families can overlap. The benchmark is
selected to expose counting errors rather than represent their natural prevalence.

Data availability and per-image terms are tracked separately from this code release;
see [data licensing](DATA_LICENSE.md).
