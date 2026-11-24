# Benchmark notes

## How the comparison was made
- **Same split:** every row uses the official CUB-200-2011 test set (5,794 images); our models were selected on our validation split only, never on the test set.
- **Same metric:** top-1 accuracy, the number all the cited papers report. Our other metrics (macro F1, top-5) are in `outputs/comparison/`.
- **Published numbers are cited, not re-run.** Each has a source and URL in `configs/benchmark.yaml`; check them against the papers before the report.

## Results

| Model | Top-1 accuracy | Input | Source |
|---|---|---|---|
| TransFG | 91.7% | 448px | He et al., AAAI 2022 |
| ViT-B/16 fine-tuned | 90.3% | 448px | He et al., "TransFG: A Transformer Architecture for Fine-grained Recognition", AAAI 2022 - comparison table |
| NTS-Net | 87.5% | 448px | Yang et al., ECCV 2018 |
| ResNet-50 fine-tuned (baseline) | 84.5% | 448px | Yang et al., "Learning to Navigate for Fine-grained Classification" (NTS-Net), ECCV 2018 - baseline row |
| Bilinear CNN (B-CNN) | 84.1% | 448px | Lin, RoyChowdhury & Maji, "Bilinear CNN Models for Fine-grained Visual Recognition", ICCV 2015 |
| EfficientNet-B0 (224px) | 76.0% | 224px | this project |
| ResNet-18 (224px) | 67.5% | 224px | this project |
| Scratch CNN (128px) | 62.3% | 128px | this project |
| Scratch CNN (224px) | 62.1% | 224px | this project |

## Why our numbers differ
- Our best model (EfficientNet-B0 (224px)) reaches 76.0%. The plain fine-tuned ResNet-50 baseline in the literature reaches 84.5%.
- **Input resolution:** the published methods use 448x448 images; we use 224x224 (4x fewer pixels), which loses the small details that separate similar species.
- **Model size:** ResNet-50 / ViT-B are much larger than our ResNet-18, EfficientNet-B0 and the 5-block scratch CNN.
- **Training budget:** our transfer models train for only 3 + 3 epochs and fine-tune just the last stage; published baselines fine-tune the whole network for many epochs.
- **Specialised architectures:** NTS-Net and TransFG add part-localisation / patch-selection modules designed for fine-grained recognition; ViT/TransFG also start from the much larger ImageNet-21k.
- **From scratch:** the scratch CNN has no pretraining at all, so it learns everything from 4,795 images, which is why it is far below every pretrained model.

## Open-source models considered but not run
- [AshProg/bird-classifier-convnext (ConvNeXt-Base, Hugging Face)](https://huggingface.co/AshProg/bird-classifier-convnext): Train/test split and label order are not documented, so it may have been trained on our test images and its class IDs may not match ours. Weights are a full pickled model (.pth), which can run arbitrary code when loaded.
- [zwe-zwe/CUB-200 (ConvNeXt-Tiny, Hugging Face)](https://huggingface.co/zwe-zwe/CUB-200): Trained on a different, assignment-specific split (4,829 train / 1,204 test) whose overlap with the official test set is unknown, so scoring it on our test set could be inflated by test images it was trained on. Pickle-based checkpoint.
