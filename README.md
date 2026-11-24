# From Pixels to Language: Milestone 1 (CUB-200-2011 bird classification)

CSE485/CSE488 Deep Learning final project, Milestone 1: **EDA (Task 1.1)** and **image classification and
encoder training (Task 1.2)** on CUB-200-2011 (200 bird species, 5,994 training / 5,794 test images).

We train a CNN encoder **from scratch** as a baseline, apply **transfer learning** with ImageNet-pretrained
ResNet-18 and EfficientNet-B0, tune hyperparameters with **grid and random search**, track every run with
**MLflow**, explain predictions with **Grad-CAM**, compare scratch vs pretrained, and **benchmark** against
published results. All models use **224×224** inputs. Everything runs through one CLI.

---

## 1. Setup

Python 3.10–3.12. From the project root (the folder containing `src/`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli check
```

On Linux/macOS use `.venv/bin/python`. For an NVIDIA GPU, first install the CUDA build of `torch` and
`torchvision` from <https://pytorch.org/get-started/locally/>. Training is meant for a GPU: use
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) on a free Colab T4. The CUB dataset (~1.1 GB)
downloads automatically from Hugging Face the first time a command needs it.

### Trained models

Checkpoints are **not** in git (spec: Google Drive link). Download them from the team Drive folder
**<https://drive.google.com/drive/folders/1sG0Mud5-jvYyAeDz3TKZ2MTvK3r4jWFj?usp=sharing>** into `checkpoints/`:

| File | Model | Input |
|---|---|---|
| `checkpoints/efficientnet_b0_best.pt` | EfficientNet-B0, transfer learning (**final model**) | 224px |
| `checkpoints/resnet18_best.pt` | ResNet-18, transfer learning | 224px |
| `checkpoints/scratch_cnn_best.pt` | Scratch CNN (baseline) | 224px |
| `checkpoints/scratch_cnn_legacy128.pth` | Scratch CNN, original Member 2 run (legacy, for reference) | 128px |

To browse every logged run, copy the team's `mlflow.db` from the Drive folder to `outputs/mlflow.db` and run
`mlflow ui` (see Commands).

Any checkpoint loads through `src/models/registry.py`, which reads the image size and class names from the
file, so `predict`, `gradcam` and `evaluate` work with all of them.

---

## 2. Commands

Every command: `python -m src.cli <command> [--task classification] [--config FILE] [--set key=value ...]`.
`--set` overrides any config value, e.g. `--set training.epochs=5 --set data.num_workers=2`.
Run `python -m src.cli <command> --help` for all options.

| Step | Command | Output |
|---|---|---|
| Sanity check (offline) | `python -m src.cli check` | console |
| **EDA** (Task 1.1) | `python -m src.cli eda` | `outputs/eda/` |
| Augmentation preview | `python -m src.cli augment-preview` | `outputs/eda/06_augmentation_*.png` |
| **Tune** scratch CNN | `python -m src.cli tune --config configs/tuning_scratch_cnn.yaml` | `outputs/tuning/scratch_cnn/` |
| **Tune** ResNet-18 | `python -m src.cli tune --config configs/tuning_resnet18.yaml` | `outputs/tuning/resnet18/` |
| **Train** scratch CNN | `python -m src.cli train --config configs/scratch_cnn.yaml` | `checkpoints/scratch_cnn_best.pt`, `outputs/training/` |
| **Train** ResNet-18 | `python -m src.cli train --config configs/resnet18.yaml` | `checkpoints/resnet18_best.pt` |
| **Train** EfficientNet-B0 | `python -m src.cli train --config configs/efficientnet_b0.yaml` | `checkpoints/efficientnet_b0_best.pt` |
| **Evaluate** on the test set | `python -m src.cli evaluate --model all` | `outputs/evaluation/<model>/` |
| **Compare** scratch vs transfer | `python -m src.cli compare` | `outputs/comparison/` |
| **Benchmark** vs published | `python -m src.cli benchmark` | `outputs/benchmark/` |
| **Predict** one image | `python -m src.cli predict --checkpoint checkpoints/efficientnet_b0_best.pt --image bird.jpg --top-k 5` | console / `--output` JSON |
| **Grad-CAM** one image | `python -m src.cli gradcam --checkpoint checkpoints/efficientnet_b0_best.pt --image bird.jpg` | `outputs/gradcam/` |
| Experiment tracking | `mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db` | <http://127.0.0.1:5000> |
| Tests | `python -m pytest -q` | console |

The final models were trained **with the default settings in each config** (the `train` commands above). Add
`--use-tuned` to train with the best settings of the last `tune` run instead
(`outputs/tuning/<model>/best_params.yaml`); in our full-length runs those scored lower (see Results).
On Windows with an NVIDIA GPU, add `--set data.num_workers=4` to speed up data loading.

Quick CPU smoke test of training (a few minutes): `python -m src.cli train --config configs/resnet18.yaml
--set training.head_epochs=1 --set training.finetune_epochs=0 --set data.max_train_images=400
--set data.max_val_images=200 --set training.checkpoint_name=smoke.pt --no-mlflow`

---

## 3. Results

Official CUB-200-2011 test set, 5,794 images, all models at 224×224, trained on an NVIDIA RTX 5060
(`evaluate` / `compare` / `benchmark`). Checkpoints were chosen on validation accuracy only.

| Model | Input | Test acc | Top-5 | Macro F1 | Val acc | Training |
|---|---|---|---|---|---|---|
| Scratch CNN | 224px | 62.06% | 86.12% | 61.63% | 58.72% | 80 epochs, 31 min |
| ResNet-18, transfer | 224px | 67.50% | 91.35% | 67.24% | 64.72% | 3 + 3 epochs, 1.3 min |
| **EfficientNet-B0, transfer** (final) | 224px | **75.99%** | **94.65%** | **75.76%** | **73.56%** | 3 + 3 epochs, 1.8 min |
| *Scratch CNN, legacy checkpoint* | *128px* | *62.34%* | *86.00%* | *61.93%* | *60.05%* | *reference only* |

- **Pretraining helps:** EfficientNet-B0 beats the scratch CNN by 14 points of accuracy while training for
  6 epochs instead of 80. Going from 128px to 224px did not improve the scratch CNN, which still underfits.
- **Tuning:** random search found a better setting than grid search for both the scratch CNN (9.59% vs 8.09%
  after 15-epoch trials) and ResNet-18 (42.87% vs 42.04% after 2-epoch trials), with the same budget. In
  full-length training the **default settings won** (scratch 58.72% vs 55.21%, ResNet-18 64.72% vs 61.22%
  validation accuracy): short trials favour higher or lower learning rates than the full run needs, and
  2-epoch head-only trials cannot show the benefit of weight decay. The final models therefore use the
  default configs (`outputs/tuning/<model>/` keeps every trial and the grid-vs-random comparison).
- **Confusions** sit between near-identical species of the same family (gulls, sparrows, terns, flycatchers,
  warblers). EfficientNet-B0's most frequent: Chipping → Tree Sparrow (13 of 30), Elegant → Caspian Tern (11),
  California → Western Gull (10), Chuck-will's-widow → Whip-poor-will (10), Common → Arctic Tern (10)
  (`outputs/evaluation/<model>/confusion_matrix.png` and `most_confused_pairs.csv`).
- **Grad-CAM** (`outputs/evaluation/<model>/gradcam_correct_vs_wrong.png`): correct predictions focus on the
  bird's head, bill and plumage pattern, never the background. Several confident "errors" are **label
  errors in the test set**: images labelled Chipping Sparrow, Green Kingfisher and Red-headed Woodpecker
  show a Lazuli Bunting, a Pied Kingfisher and a Red-bellied Woodpecker, exactly what the models predict.
- **Benchmark** (`outputs/benchmark/`): published methods reach 84.1–91.7% using 448px inputs and larger or
  specialised backbones; `benchmark_notes.md` explains the gap.

The full write-up (EDA, experiments, tuning, MLflow screenshots, Grad-CAM analysis) is in the Milestone 1 report.

---

## 4. Milestone 1 requirements → where they are

| Requirement (project PDF) | Where |
|---|---|
| Load CUB-200-2011 | `src/data/hf.py` |
| EDA: class distribution, image sizes/quality, caption length/vocabulary/typos, missing data, duplicates, corrupted files | `src/data/eda.py`, `audit.py`, `captions.py` → `outputs/eda/` |
| Clean captions (split the 10 descriptions) | `src/data/captions.py` → `outputs/eda/captions/captions_clean_*.jsonl` |
| Train/validation split | `src/data/split.py`, `data/splits/train_validation_split.csv` (80/20 stratified, fixed) |
| Preprocessing + augmentation, with justification | `src/data/transforms.py`, [`docs/preprocessing_decisions.md`](docs/preprocessing_decisions.md) |
| Augmentation visualisations | `outputs/eda/06_augmentation_*.png` |
| Reusable PyTorch DataLoaders | `src/data/dataset.py`, `src/data/loaders.py` (verified batch by batch in `eda`) |
| CNN encoder from scratch + 200-class head | `src/models/bird_cnn.py` |
| Training and validation loops (reusable) | `src/training/engine.py`, `trainer.py`, `strategies.py` |
| Experiment tracking | MLflow, `src/utils/tracking.py` → `outputs/mlflow.db` |
| Grid Search + Random Search, compared | `src/training/tuning.py` → `outputs/tuning/<model>/grid_vs_random.{csv,png}` |
| Training/validation loss and accuracy curves | `outputs/training/<run>/learning_curves.png` |
| Pretrained CNN with new 200-class head, fine-tuned | `src/models/transfer.py` (ResNet-18, EfficientNet-B0; head stage then fine-tune) |
| Accuracy, precision, recall, F1, confusion matrix | `src/evaluation/classification.py` → `outputs/evaluation/<model>/` |
| Grad-CAM on correct and wrong predictions | `src/visualization/gradcam.py` → `outputs/evaluation/<model>/gradcam_correct_vs_wrong.png` |
| Scratch vs pretrained comparison | `python -m src.cli compare` → `outputs/comparison/` |
| Benchmark vs published results | `configs/benchmark.yaml`, `src/evaluation/benchmark.py` → `outputs/benchmark/` |
| CLI for training and inference | `src/cli.py` |
| Notebooks only for experiments | `notebooks/` (the final code never depends on them) |

---

## 5. Project structure

```
configs/        data.yaml (shared settings) + one YAML per model and per tuning search + benchmark.yaml
data/splits/    the fixed team train/validation manifest
src/
  cli.py        single entry point
  data/         hf loading, split, dataset, transforms, loaders, audit, captions, eda
  models/       bird_cnn (scratch), transfer (ResNet-18 / EfficientNet-B0), registry (build/load any model)
  training/     engine (one loop), strategies (scratch / transfer), trainer, tuning
  evaluation/   classification metrics, evaluate + compare, benchmark
  visualization/ plots, EDA figures, augmentation previews, Grad-CAM
  utils/        config (YAML inheritance + overrides), seeding, MLflow tracking
tests/          pytest suite (offline, uses a tiny fake dataset)
notebooks/      experiment records + Colab training notebook
outputs/        generated figures/metrics (legacy/ keeps the original 128px Member 2 results)
checkpoints/    trained models (gitignored; Google Drive)
```

Design: every model goes through the same dataset, loaders, training engine and evaluation code. A model is
chosen by name in its config (`model.architecture`), and its training recipe by `training.strategy`, so a
new encoder can be swapped in without changing the training loop (spec 5.2).

---

## 6. Known limitations

- **Single runs:** every configuration was trained once (seed 42), so differences of 1–2 points are within
  run-to-run noise. The validation split has only about 6 images per species.
- **Tuning trials** were shorter than the final training and did not include the fine-tune stage of the
  transfer models; the fine-tune learning rate was not tuned.
- **Test labels** contain some errors (see Grad-CAM above); the official labels are kept so the benchmark
  stays comparable.
- **299 flagged images** (train + test: small, dark, low-contrast or grey-scale) are listed for manual review
  in `outputs/eda/review/`; none were removed automatically.
- Training on a CPU is impractical: one scratch-CNN epoch on just 400 images took about 3 minutes on a laptop CPU.
- **Published benchmark numbers** in `configs/benchmark.yaml` were collected from the papers' arXiv versions
  and must be double-checked by the team; they use 448px inputs and larger backbones (see
  `outputs/benchmark/benchmark_notes.md`).

## 7. Team

- **Member 1, Youssef Mohamed Abboud (23P0062)** - Dataset & EDA (Task 1.1): data loading, EDA, caption cleaning, split, preprocessing, DataLoaders.
- **Member 2, Youssef Mohamed ElSayed (23P0215)** - CNN from scratch (Task 1.2): encoder architecture, training/validation loops, tracking, tuning, curves.
- **Member 3, Mohamed Ahmed Eissa (23P0143)** - Transfer learning & evaluation (Task 1.2): ResNet-18/EfficientNet-B0, metrics, Grad-CAM, comparison, benchmark.
