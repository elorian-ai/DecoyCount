# Third-party dependencies

This repository contains the standalone Python evaluator. It does not vendor
dependencies or include SAM3, lmms-eval, model weights, or internal agent services.
Installing requirements downloads packages governed by their own licences; this
repository's MIT licence does not replace those terms.

Direct dependencies:

| Package | Upstream licensing information |
| --- | --- |
| openai | [openai/openai-python](https://github.com/openai/openai-python) |
| pandas | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| numpy | [numpy/numpy](https://github.com/numpy/numpy) |
| Pillow | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| pillow-heif | [bigcat88/pillow_heif](https://github.com/bigcat88/pillow_heif) |

HEIC support can involve bundled native libraries with additional terms. SVG
preparation uses an external `rsvg-convert` installation. Review those distributions
and the resolved transitive dependencies before bundling a container or binary.
This source-only inventory is not a completed dependency clearance or a permission
to redistribute dataset images.
