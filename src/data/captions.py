"""Caption analysis and cleaning (Member 1, Task 1.1; used for captioning in Milestone 2).

The ``description`` field holds ten captions separated by new lines. Cleaning steps:
1. split on new lines;
2. normalise whitespace and lower-case (captions are mostly lower-case already);
3. drop exact repeats of a caption for the same image;
4. drop fragments with fewer than ``MIN_WORDS`` words (e.g. the lone caption "this");
5. keep the original text untouched in the dataset; cleaned captions are written to JSONL.
Typos are *reported* (rare words very similar to frequent ones) but not auto-corrected,
because bird vocabulary contains many rare-but-correct words (e.g. "superciliary").
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

MIN_WORDS = 3
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOPWORDS = set('a an the and or of with has have is are this that its it to in on at as for bird birds very '
                'which while from by be has'.split())


def split_captions(description) -> list[str]:
    return [re.sub(r'\s+', ' ', line).strip() for line in str(description or '').splitlines() if line.strip()]


def clean_captions(raw: list[str]) -> tuple[list[str], dict]:
    seen, kept = set(), []
    stats = {'duplicates_removed': 0, 'fragments_removed': 0}
    for caption in raw:
        text = caption.lower()
        if len(WORD.findall(text)) < MIN_WORDS:
            stats['fragments_removed'] += 1
            continue
        if text in seen:
            stats['duplicates_removed'] += 1
            continue
        seen.add(text)
        kept.append(text)
    return kept, stats


def analyse_captions(ds, split_name: str, out_dir) -> dict:
    """Clean every image's captions, save JSONL, and return statistics and per-caption tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_caption, per_image, issues = [], [], []
    raw_tokens, clean_tokens = Counter(), Counter()
    totals = {'raw_captions': 0, 'clean_captions': 0, 'duplicates_removed': 0, 'fragments_removed': 0}
    jsonl = out_dir / f'captions_clean_{split_name}.jsonl'
    descriptions, names, labels = ds['description'], ds['file_name'], ds['label']
    with jsonl.open('w', encoding='utf-8') as handle:
        for i, (description, file_name, label) in enumerate(zip(descriptions, names, labels)):
            raw = split_captions(description)
            clean, stats = clean_captions(raw)
            for caption in raw:
                raw_tokens.update(WORD.findall(caption.lower()))
            for caption in clean:
                words = WORD.findall(caption)
                clean_tokens.update(words)
                per_caption.append({'row_index': i, 'words': len(words), 'characters': len(caption)})
            per_image.append({'row_index': i, 'raw_captions': len(raw), 'clean_captions': len(clean)})
            if len(raw) != 10:
                issues.append({'row_index': i, 'issue': 'not ten captions', 'detail': len(raw)})
            if stats['duplicates_removed']:
                issues.append({'row_index': i, 'issue': 'repeated caption', 'detail': stats['duplicates_removed']})
            if stats['fragments_removed']:
                issues.append({'row_index': i, 'issue': 'fragment caption', 'detail': stats['fragments_removed']})
            totals['raw_captions'] += len(raw)
            totals['clean_captions'] += len(clean)
            totals['duplicates_removed'] += stats['duplicates_removed']
            totals['fragments_removed'] += stats['fragments_removed']
            handle.write(json.dumps({'row_index': i, 'file_name': file_name, 'label': int(label),
                                     'captions': clean}, ensure_ascii=False) + '\n')
    captions = pd.DataFrame(per_caption)
    images = pd.DataFrame(per_image)
    pd.DataFrame(issues, columns=['row_index', 'issue', 'detail']).to_csv(
        out_dir / f'caption_issues_{split_name}.csv', index=False)
    summary = {**totals, 'images': len(images),
               'vocabulary_raw': len(raw_tokens), 'vocabulary_clean': len(clean_tokens),
               'words_per_caption': captions['words'].describe().round(2).to_dict(),
               'captions_per_image_after_cleaning': images['clean_captions'].value_counts().sort_index().to_dict(),
               'words_seen_once': sum(1 for c in clean_tokens.values() if c == 1),
               'images_with_issues': len({x['row_index'] for x in issues})}
    return {'summary': summary, 'captions': captions, 'images': images, 'tokens': clean_tokens}


def typo_candidates(tokens: Counter, rare_max=2, common_min=20, cutoff=0.85) -> pd.DataFrame:
    """Rare words that look like misspellings of frequent words (e.g. 'whte' -> 'white')."""
    common = [w for w, c in tokens.items() if c >= common_min and len(w) > 3]
    rows = []
    for word, count in tokens.items():
        if count > rare_max or len(word) <= 3:
            continue
        match = difflib.get_close_matches(word, common, n=1, cutoff=cutoff)
        if match and match[0] != word:
            rows.append({'word': word, 'count': count, 'likely_meant': match[0], 'meant_count': tokens[match[0]]})
    return pd.DataFrame(rows, columns=['word', 'count', 'likely_meant', 'meant_count']).sort_values(
        'meant_count', ascending=False)


def top_words(tokens: Counter, n=30, drop_stopwords=True) -> list[tuple[str, int]]:
    return [(w, c) for w, c in tokens.most_common() if not (drop_stopwords and w in STOPWORDS)][:n]
