
# Bird Species Classification — Deep Learning Project

## Overview

This project classifies 200 bird species from the CUB-200-2011
dataset using a convolutional neural network trained from scratch.

## Team Contributions

- Member 1: Dataset preparation, EDA and preprocessing
- Member 2: Scratch CNN, training and validation
- Member 3: Transfer learning, evaluation and Grad-CAM

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

```bash
python -m src.cli train --task classification --config configs/scratch.yaml
```

## Prediction

```bash
python -m src.cli predict --task classification --checkpoint outputs/scratch_cnn/best_model.pth --image path/to/bird.jpg
```

## Outputs

Training saves the best checkpoint, training history,
loss plot and accuracy plot under outputs/scratch_cnn/.
