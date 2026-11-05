# Smart Inspection Edge Platform

Industrial anomaly detection built to understand the methods, not just call a library: PaDiM and STFPM re-implemented from scratch in PyTorch and benchmarked head to head on MVTec AD using image and pixel AUROC. Next steps: packaging as a service (FastAPI, Streamlit, Docker), then edge inference on a Jetson Orin Nano with ONNX and TensorRT. It is the first step of a longer path toward real-time perception for robotics.

## Status (October 2026)

**Phase 1: Software Platform (in progress)**
- [x] MVTec AD data pipeline (`AnomalyDataset`, lazy loading, binary masks)
- [x] PaDiM and STFPM from scratch, checked against their papers
- [x] Training, evaluation (image and pixel AUROC, per defect type), benchmark (FPS, VRAM)
- [x] Anomaly map visualization, CI (ruff + pytest)
- [ ] FastAPI service (`/predict`, `/health`, `/metrics`), decision threshold, Precision/Recall/F1
- [ ] Streamlit dashboard, Docker, automatic PDF report
- [ ] Human feedback loop, degraded conditions (lighting, occlusion, sensor noise)

**Phase 2: Edge AI deployment (planned)**: ONNX export, TensorRT (C++ API), FP16/INT8, Jetson Orin Nano.

## Results (MVTec AD, test set, GPU, seed 42)

| | bottle image AUROC | bottle pixel AUROC | cable image AUROC | cable pixel AUROC |
|---|---|---|---|---|
| STFPM | 1.000 | 0.988 | 0.914 | 0.947 |
| PaDiM | 0.994 | 0.983 | 0.852 | 0.974 |

| Cable | STFPM | PaDiM |
|---|---|---|
| Throughput (batch 1) | ~456 FPS | ~582 FPS |
| Resident VRAM | ~102 MB | ~375 MB |
| VRAM per inference | ~10 MB | ~13 MB |

Measured on one machine (RTX 3070 Ti). Throughput is 1000 / median latency over 60 single-image predictions (after 10 warm-up runs, host-to-device transfer included). Image preprocessing (resize, normalization) is excluded.

The per-defect AUROC on cable is lowest for `cable_swap` (STFPM 0.736, PaDiM 0.716), an inconclusive result under the decision rule written beforehand (0.65 to 0.8 = inconclusive, see the technical log).

Anomaly maps (cable): [outputs/figures/figure_20261004_123157/comparison/](outputs/figures/figure_20261004_123157/comparison/) shows PaDiM and STFPM maps on the same images (3 best-detected and 3 most-missed anomalies per method, color scale shared per method).

## Known limits

- Only `bottle` and `cable` are evaluated (out of 15 MVTec categories), one training run per method.
- No decision threshold yet: results are threshold-free AUROC.
- Results can differ slightly between CPU and GPU, even with a fixed seed.
- Checkpoints are not versioned (`*.pt` is ignored): `scripts/benchmark_fps_vram.py`, `scripts/compare_method_bysubcategory.py` and `scripts/generate_*.py` read checkpoint names written as constants at the top of each script.

## Project layout

```text
.
├── configs/                 # common.yaml, padim.yaml, stfpm.yaml
├── docs/
│   └── TECHNICAL_LOG.md     # design decisions, bugs, results by session
├── outputs/
│   ├── figures/             # anomaly map figures (selected runs versioned)
│   ├── models/              # checkpoints (*.pt, not versioned)
│   └── reports/             # JSON reports (AUROC, benchmark, diagnostics)
├── scripts/                 # comparison, benchmark, figures (debug/: step-by-step checks)
├── src/smart_inspection/
│   ├── api/                 # FastAPI service (Phase 1, not started)
│   ├── config/              # YAML loading and path resolution
│   ├── data/                # AnomalyDataset
│   ├── evaluation/          # AUROC, FPS/VRAM benchmark, reference checks
│   ├── io/                  # run ids, JSON, figures
│   ├── models/              # AnomalyMethod (abstract), factory, padim/, stfpm/
│   ├── training/            # generic train() orchestrator
│   └── visualization/       # anomaly map figures
└── tests/                   # pytest (integration tests need the dataset and pretrained weights)
```

## Setup

Python 3.12 or higher and [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv downloads a matching Python if none is installed).

```bash
git clone https://github.com/dhesdin/smart-inspection-edge-platform.git
cd smart-inspection-edge-platform
make setup
source .venv/bin/activate
```

`make setup` runs `uv sync --locked` with every extra and the CUDA 13.2 PyTorch build (workstation). For a smaller target, pass the extras you need: `make setup EXTRAS=serve,edge`, or `make setup EXTRAS=train,cpu,dev` without an NVIDIA GPU.

Dependencies are split by target. The core (`numpy`, `pyyaml`, `pillow`) is always installed.

| Extra | Target | Contents |
|---|---|---|
| `train` | Workstation with NVIDIA GPU | torch, torchvision, scipy, scikit-learn, matplotlib |
| `export` | ONNX export (install with `train`, needs torch) | onnx |
| `serve` | Inference API | fastapi, uvicorn, python-multipart |
| `dashboard` | Dashboard | streamlit, matplotlib |
| `edge` | Edge runtime, no torch | onnxruntime (CPU; on Jetson, the GPU runtime comes from JetPack) |
| `dev` | Lint and tests | ruff, pytest, httpx |
| `cpu` / `cu132` | PyTorch build, with `train` (exactly one) | torch and torchvision from the CPU-only or CUDA 13.2 PyTorch index |

Every target installs from the same `uv.lock` (exact versions for all platforms). After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`.

Windows:

```bash
uv sync --locked --extra train --extra cu132 --extra export --extra serve --extra dashboard --extra edge --extra dev
.venv\Scripts\activate
```

## Dataset

Download MVTec AD from the [official site](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and place it in `datasets/mvtec_anomaly_detection/`. The dataset is not included in this repository and is licensed CC BY-NC-SA 4.0 (non-commercial use, attribution required).

## Usage

```bash
python scripts/compare_padim_stfpm.py            # trains both methods, saves checkpoints and a JSON report
python scripts/benchmark_fps_vram.py             # FPS and VRAM per method
python scripts/compare_method_bysubcategory.py   # AUROC per defect type, checked against a reference report
python scripts/generate_anomaly_figures.py       # anomaly maps: best-detected, most-missed, highest-scored normals
python scripts/generate_cable_swap_figures.py    # cable_swap: maps, score diagnostic, argmax-in-mask test
make lint        # ruff check + format check
make format      # ruff fix + format
make test        # all tests, including those that need the dataset or pretrained weights
```

`compare_padim_stfpm.py` trains on `cable`: set `category_dataset` in the script to `bottle` for the bottle results. The other scripts load existing checkpoints and reports: edit the constants at the top of each script (`CHECKPOINTS`, `REFERENCE_REPORT`, ...).

CI runs two jobs: `lint` (ruff, core + `dev` only) and `test` (`pytest -q -m "not integration"` with CPU-only torch, no dataset needed).

## References

- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, 2020 (arXiv:2011.08785)
- Wang et al., *Student-Teacher Feature Pyramid Matching for Anomaly Detection*, BMVC 2021 (arXiv:2103.04257)
- Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019

## License

Code: MIT. Dataset: MVTec AD, CC BY-NC-SA 4.0.