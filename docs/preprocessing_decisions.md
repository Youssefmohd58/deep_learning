# Preprocessing decisions (Task 1.1)

These decisions follow from the EDA (`python -m src.cli eda`, figures and numbers in `outputs/eda/`).
The numbers quoted here come from `outputs/eda/eda_summary.json`.

## What the data looks like

| Finding | Value | Consequence |
|---|---|---|
| Species | 200 | 200-way classifier head |
| Images per species (train) | 29–30 (mean 29.97, max/min ratio 1.03) | The training set is balanced, so there is no class weighting or resampling. Accuracy is a fair headline metric; macro F1 is also reported. |
| Images per species (test) | 11–30 | The test set is slightly unbalanced, so macro *and* weighted F1 are both reported. |
| Image size | width mostly 500px (min 121), height 120–500px (median 375) | Every image is resized; the median photo is about twice the 224px model input. |
| Smaller than 224px | 148 images | They are upscaled to 224 (slightly blurry); flagged for review, not dropped. |
| Grey-scale (mode L) | 8 images | Converted to 3-channel RGB before every transform. |
| Very dark / low contrast | 33 / 110 images | Flagged for review with contact sheets; kept, because real photos vary in lighting. |
| Corrupted files | 0 | Nothing to remove. |
| Missing labels / captions / file names | 0 in both splits | No imputation needed. |
| Exact duplicate images | 1 group, both copies in the same split | No leakage between train, validation and test. |
| Captions per image | 10 raw; 59,940 train captions | 639 repeated captions and 1 fragment ("this") were removed; 5,410 of 5,994 images keep all 10. |
| Words per caption | mean 15.3, median 14, max 66 | Matters for the caption decoder length in Milestone 2. |
| Vocabulary | 4,067 word types; 1,867 appear once | 508 rare words look like typos of common words (e.g. "whte" → "white"); they are reported, not auto-corrected (see below). |

## Decisions

1. **Train/validation split: 80/20, stratified by species, seed 42, saved once.**
   The official test split is held out completely and used only for the final evaluation. The validation
   split (1,199 images, about 6 per species) is used for model selection and tuning. The manifest
   `data/splits/train_validation_split.csv` is fixed and its SHA-256 is stored in every checkpoint, so all
   models are trained and compared on exactly the same images. Stratification keeps every species in both
   splits, which plain random splitting would not guarantee with only 30 images per species.

2. **One input size for all models: 224×224.** It matches the ImageNet pretraining resolution of ResNet-18
   and EfficientNet-B0, and keeps more fine detail (beak shape, eye rings, wing bars) than 128px. The scratch
   CNN's last feature map grows from 4×4 to 7×7, which also gives sharper Grad-CAM maps. Using one size for
   every model makes the scratch-vs-pretrained comparison differ only in pretraining. The original 128px
   scratch checkpoint is kept as a clearly labelled *legacy* model until its 224px retrain finishes.

3. **Evaluation preprocessing (all models):** convert to RGB, resize the shorter side to 256, center-crop
   224, scale to [0, 1], normalise with the ImageNet mean/std. The 256→224 crop is the standard ImageNet
   protocol; birds are usually centred, so the crop rarely cuts them. The ImageNet statistics are required by
   the pretrained models and also work well for the scratch CNN.

4. **Training augmentation: two recipes, chosen per model in its config.**
   - `transfer_mild` (ResNet-18, EfficientNet-B0): random resized crop keeping 80–100% of the image,
     horizontal flip, ±10° rotation, ±10% brightness/contrast. Pretrained features already generalise, so
     gentle augmentation is enough, and heavy distortion mainly slows fine-tuning.
   - `scratch_strong` (scratch CNN): random resized crop keeping 35–100% (zoom in on parts of the bird),
     horizontal flip, TrivialAugmentWide (one random photometric/geometric operation of random strength),
     and random erasing (p = 0.25, simulates occlusion by branches). A network with no prior knowledge
     memorises 4,795 images quickly; strong augmentation is the main defence against overfitting.
   - Not used: vertical flips (birds are almost never upside down) and strong colour changes in the mild
     recipe (colour is a key feature for telling species apart).
   - Previews: `outputs/eda/06_augmentation_*.png`. In the preview, erased patches look black; in training
     they are filled with the average ImageNet colour, because erasing runs after normalisation.

5. **Captions (for Milestone 2):** split on new lines, normalise whitespace, lower-case, remove exact repeats
   within an image and fragments shorter than three words. The original text is never modified; cleaned
   captions are written to `outputs/eda/captions/captions_clean_{train,test}.jsonl`. **Typos are reported,
   not corrected,** because the rare-word list also contains correct but uncommon bird terms; automatic
   correction could damage them. The list is in `typo_candidates_train.csv` for a later decision (for
   example, a hand-checked correction map before building the caption vocabulary).

6. **Images flagged for review are kept.** 299 images (train + test) are flagged as small, dark, low-contrast
   or grey-scale and listed with contact sheets in `outputs/eda/review/`. Removing training images needs a
   person to decide, and test images must never be removed, so the benchmark stays comparable.

7. **DataLoaders:** one `CUBClassificationDataset` for every model. It decodes images lazily, returns the
   image, its label and its original row index (so every prediction can be traced to its file), and uses a
   seeded shuffle for reproducibility. `eda` iterates every batch of the train, validation and test loaders
   and checks shape (3×224×224), finite values, label range, and that no image is missing or repeated.
