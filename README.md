# DecoyCount

**Evaluating Visual-Reasoning Misalignment in Counting**

[Paper](https://arxiv.org/abs/2610.34855) · [PDF](https://arxiv.org/pdf/2610.34855) · [Dataset](https://huggingface.co/datasets/elorian-ai/DecoyCount)

DecoyCount pairs 1,660 real-world counting items with pre-registered decoys and three
prompt conditions: **neutral**, **alignment-cued**, and **multi-hop**. The multi-hop
condition adds scenario context and, where applicable, a second image. The
reasoning-alignment gap is cued minus neutral accuracy, in percentage points;
it measures the improvement under a matched cue, not every source of reasoning error.

This repository contains the standalone API evaluator, image preparation and scorer.
Dataset publication is separate: the Hugging Face link is the release destination,
not a claim that the complete download is already public. Images, consent records
and historical model outputs are not included here.

## Evaluate

Requires Python 3.10+. Once the release is available, obtain its canonical layout:
`data/index.csv`, `data/tasks/Task-<id>/image/`, and optional `image_secondary/`.
See [dataset format and limitations](DATASET.md).

```bash
pip install -r requirements.txt
python eval/harness/normalize_images.py --repo data --outdir images

# Supply your own OpenAI-compatible endpoint and credentials.
export GATEWAY_BASE_URL=https://your-endpoint.example/v1
export GATEWAY_API_KEY=your-key
python eval/harness/run_counting.py --repo data --imgdir images \
  --models model=gateway:your-model --conditions neutral cued \
  --effort none --workers 4 --out results/my-run
python eval/harness/score.py results/my-run --out results/my-run/scored.csv
```

Use `--conditions contextual` for multi-hop (`contextual` is the legacy CLI name).
Known potentially stale derived answers require care; see [limitations](DATASET.md#limitations).
Run `--list-providers` for other configured providers; endpoint compatibility varies.
Use a new output directory for each model/configuration; an existing directory resumes a run.

## Tests

```bash
python -m unittest discover -s eval/harness -p 'test_*.py'
```

Tests use synthetic images and a local mock API; they do not call paid providers.
HEIC support uses `pillow-heif`; SVG preparation additionally requires `rsvg-convert`.

## Licensing and citation

Evaluator code: [MIT](LICENSE). Dataset images retain their **individual source
licences**; MIT does not cover them. See [data licensing](DATA_LICENSE.md) and
[dependency notices](THIRD_PARTY_NOTICES.md).

Please cite [DecoyCount, arXiv:2610.34855](https://arxiv.org/abs/2610.34855).
Report issues here using task IDs; do not post personal information or consent forms.
