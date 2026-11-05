# DecoyCount

**Evaluating Visual-Reasoning Misalignment in Counting**

[Paper](https://arxiv.org/abs/2610.34855) · [PDF](https://arxiv.org/pdf/2610.34855) · [Download images](https://github.com/elorian-ai/DecoyCount/releases/tag/v6-media-preview)

The current DecoyCount v6 release pairs **1,586 counting tasks** with pre-registered decoys and three
prompt conditions: **neutral**, **alignment-cued**, and **multi-hop**. The multi-hop
condition adds scenario context and, where applicable, a second image. The
reasoning-alignment gap is cued minus neutral accuracy, in percentage points;
it measures the improvement under a matched cue, not every source of reasoning error.

The **v6 media preview** contains **2,108 decoys and 2,369 images**
(1,586 primary + 783 secondary), with prompts and answers for all three conditions.
The paper evaluated an earlier 1,660-task bank; its results are not scores on this
edited release. Selection and remaining limitations are in [RELEASE.md](RELEASE.md).

## Download the dataset

**The images are in these five ZIP files—not in Git's `dataset/` directory or
GitHub's automatic source-code ZIP. Download all five parts:**

[Part 1](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/decoycount-v6-part-01.zip) ·
[Part 2](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/decoycount-v6-part-02.zip) ·
[Part 3](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/decoycount-v6-part-03.zip) ·
[Part 4](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/decoycount-v6-part-04.zip) ·
[Part 5](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/decoycount-v6-part-05.zip) ·
[SHA256SUMS](https://github.com/elorian-ai/DecoyCount/releases/download/v6-media-preview/SHA256SUMS)

Approximately 3.9 GB total. Part 1 also contains the evaluator, prompts, answers,
Attribution v6 and manifests; the other parts contain additional individual image
files. Save all downloads together, verify them and extract into one new directory:

```bash
shasum -a 256 -c SHA256SUMS
for part in decoycount-v6-part-*.zip; do unzip -n "$part" -d DecoyCount; done
cd DecoyCount
```

Only tasks with TRUE permission flags for every supplied image are included;
AI-generated images are retained where v6 permits them. Five primary records
still have incomplete licence/credit fields. See [data terms](DATA_LICENSE.md)
and [known limitations](RELEASE.md#remaining-metadata-limitations); publication is
not blanket rights clearance. Private consent records and historical outputs are
not distributed.

## Evaluate

Requires Python 3.10+. After extracting the ZIPs, the layout is
`dataset/index.csv`, `dataset/tasks/Task-<id>/image/`, and optional `image_secondary/`.
See [dataset format and limitations](DATASET.md).
Install `rsvg-convert` for the SVG image before preparation: `brew install librsvg`
on macOS, or `sudo apt-get install librsvg2-bin` on Ubuntu/Debian. HEIC support is
provided by `pillow-heif` in `requirements.txt`.

```bash
pip install -r requirements.txt
python eval/harness/check_release.py --repo dataset --media-only
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
The media-only check verifies files and metadata, not permission clearance. The
default full check also reports the incomplete licence/credit fields; no retained
image has a FALSE permission flag.
Run `--list-providers` for other configured providers; endpoint compatibility varies.
Use a new output directory for each model/configuration; an existing directory resumes a run.

## Tests

```bash
python -m unittest discover -s eval/harness -p 'test_*.py'
```

Tests use synthetic images and a local mock API; they do not call paid providers.

## Licensing and citation

Evaluator code: [MIT](LICENSE). Dataset images retain their **individual source
licences**; MIT does not cover them. See [data licensing](DATA_LICENSE.md) and
[dependency notices](THIRD_PARTY_NOTICES.md).

Please cite [DecoyCount, arXiv:2610.34855](https://arxiv.org/abs/2610.34855).
Report issues here using task IDs; do not post personal information or consent forms.
