"""Image audit for EDA (Member 1, Task 1.1): sizes, quality, corrupted files, duplicates, missing fields.

Every image is decoded once; the same pass records its size, brightness, contrast and a
hash of its decoded pixels (for exact-duplicate detection across train/validation/test).
Flags are for human review only; nothing is dropped automatically.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageStat
from tqdm.auto import tqdm

MIN_SIDE = 224          # smaller images must be upscaled for 224px inputs
DARK_MEAN = 40          # mean grey level (0-255) below this looks very dark
LOW_CONTRAST_STD = 15   # grey-level standard deviation below this looks flat / washed out


def scan_images(ds, split_name: str) -> pd.DataFrame:
    rows = []
    for i in tqdm(range(len(ds)), desc=f'Scanning {split_name} images'):
        row = {'split': split_name, 'row_index': i, 'decode_ok': False}
        try:
            image = ds[i]['image']
            image.load()
            width, height = image.size
            grey = image.convert('L').resize((64, 64))
            stat = ImageStat.Stat(grey)
            rgb = image.convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            row.update(decode_ok=True, width=width, height=height, aspect_ratio=width / height, mode=image.mode,
                       mean_brightness=stat.mean[0], brightness_std=stat.stddev[0], pixel_sha256=digest)
        except Exception as error:  # corrupted file: record it instead of crashing the audit
            row['error'] = f'{type(error).__name__}: {error}'
        rows.append(row)
    return pd.DataFrame(rows)


def quality_flags(meta: pd.DataFrame) -> pd.DataFrame:
    good = meta[meta.decode_ok].copy()
    reasons = []
    for row in good.itertuples():
        r = []
        if row.mean_brightness < DARK_MEAN:
            r.append('very dark')
        if row.brightness_std < LOW_CONTRAST_STD:
            r.append('low contrast')
        if min(row.width, row.height) < MIN_SIDE:
            r.append(f'smaller than {MIN_SIDE}px')
        if row.mode != 'RGB':
            r.append(f'mode {row.mode}')
        reasons.append('; '.join(r))
    good['flag_reason'] = reasons
    return good[good.flag_reason != ''].copy()


def contact_sheets(ds, flags: pd.DataFrame, out_dir, per_sheet=48) -> list[Path]:
    """Thumbnail sheets of flagged images so a person can review them quickly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(flags), per_sheet):
        chunk = flags.iloc[start:start + per_sheet]
        canvas = Image.new('RGB', (8 * 150, 6 * 150), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(chunk.itertuples()):
            x, y = (j % 8) * 150, (j // 8) * 150
            thumb = ds[int(row.row_index)]['image'].convert('RGB')
            thumb.thumbnail((142, 116))
            canvas.paste(thumb, (x + (150 - thumb.width) // 2, y))
            draw.text((x + 4, y + 120), f'{row.split} {row.row_index}', fill='black')
            draw.text((x + 4, y + 133), str(row.flag_reason)[:24], fill='firebrick')
        path = out_dir / f'flagged_contact_sheet_{start // per_sheet + 1:02d}.jpg'
        canvas.save(path, quality=90)
        paths.append(path)
    return paths


def duplicate_report(meta: pd.DataFrame, split_of_train_row: dict) -> dict:
    """Exact pixel duplicates; a duplicate that spans splits would leak information between them."""
    groups = defaultdict(list)
    for row in meta[meta.decode_ok].itertuples():
        where = split_of_train_row.get(int(row.row_index), 'train') if row.split == 'train' else 'test'
        groups[row.pixel_sha256].append({'source': row.split, 'row_index': int(row.row_index), 'split': where})
    duplicates = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in duplicates if len({m['split'] for m in g}) > 1]
    return {'exact_duplicate_groups': len(duplicates), 'cross_split_groups': len(cross),
            'groups': duplicates, 'cross_split': cross}


def missing_fields(ds, split_name: str) -> dict:
    """Count empty/missing values per text/label column (images are checked by scan_images)."""
    counts = {}
    for column in ('label', 'description', 'file_name', 'id'):
        if column not in ds.column_names:
            counts[column] = 'column missing'
            continue
        values = ds[column]
        counts[column] = int(sum(v is None or (isinstance(v, str) and not v.strip()) for v in values))
    return {'split': split_name, 'rows': len(ds), 'missing_or_empty': counts}
