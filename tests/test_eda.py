"""EDA and caption-cleaning tests on the fake dataset."""
from collections import Counter

from conftest import FakeHF
from src.data.captions import clean_captions, split_captions, typo_candidates
from src.data.eda import run_eda


def test_caption_cleaning_rules():
    raw = split_captions('This bird is red.\n\n  this   bird is RED. \nthis\nsmall brown bird with white wings')
    clean, stats = clean_captions(raw)
    assert clean == ['this bird is red.', 'small brown bird with white wings']
    assert stats == {'duplicates_removed': 1, 'fragments_removed': 1}


def test_typo_candidates_find_misspellings():
    tokens = Counter({'white': 500, 'wings': 300, 'whte': 1, 'superciliary': 2})
    found = typo_candidates(tokens)
    assert found.set_index('word').loc['whte', 'likely_meant'] == 'white'


def test_run_eda_writes_all_figures(tiny_cfg):
    data = {'train': FakeHF(40), 'test': FakeHF(16, seed=1)}
    summary = run_eda(tiny_cfg, data, verify_loaders=True, verbose=False)
    out = __import__('src.utils.config', fromlist=['output_dir']).output_dir(tiny_cfg, 'eda')
    for name in ('01_class_distribution.png', '02_image_sizes_and_quality.png', '03_caption_lengths.png',
                 '04_top_words.png', '05_sample_images.png', '06_augmentation_scratch_strong.png',
                 '06_augmentation_transfer_mild.png', 'eda_summary.json'):
        assert (out / name).is_file(), name
    assert summary['loader_verification']['train']['passed']
    assert summary['captions_train']['fragments_removed'] == 40
