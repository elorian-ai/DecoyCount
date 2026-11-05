# DecoyCount

**Evaluating Visual-Reasoning Misalignment in Counting**

[Paper](https://arxiv.org/abs/2610.34855) · [PDF](https://arxiv.org/pdf/2610.34855) · [Dataset](https://huggingface.co/datasets/elorian-ai/DecoyCount)

DecoyCount pairs 1,660 real-world counting items with pre-registered decoys and three
prompt conditions: **neutral**, **alignment-cued**, and **multi-hop**. The multi-hop
condition adds scenario context and, where applicable, a second image. The
reasoning-alignment gap is cued minus neutral accuracy, in percentage points;
it measures the improvement under a matched cue, not every source of reasoning error.

This repository contains the standalone API evaluator, image preparation and scorer.
The [release candidate](dataset/) contains **1,635 tasks**, including neutral,
alignment-cued and multi-hop prompts and answers, plus Attribution v5 and an image
manifest. It excludes 25 tasks from the paper's 1,660-item bank and expects 823
secondary images. **Images are not published yet:** recorded permissions still
block the complete release. See [release status](RELEASE.md). The Hugging Face
link is a destination, not confirmation of availability. Consent records and
historical model outputs remain private.

## Evaluate

Requires Python 3.10+. Once images are released, their canonical layout will be
`dataset/index.csv`, `dataset/tasks/Task-<id>/image/`, and optional `image_secondary/`.
See [dataset format and limitations](DATASET.md).

```bash
pip install -r requirements.txt
python eval/harness/check_release.py --repo dataset
python eval/harness/normalize_images.py --repo dataset --outdir images

# Supply your own OpenAI-compatible endpoint and credentials.
export GATEWAY_BASE_URL=https://your-endpoint.example/v1
export GATEWAY_API_KEY=your-key
python eval/harness/run_counting.py --repo dataset --imgdir images \
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
