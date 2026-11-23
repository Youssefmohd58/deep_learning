"""Grid and random search tests (tiny fake data, 1-epoch trials)."""
import pandas as pd

from src.training.tuning import apply_tuned, grid_trials, random_trials, run_search
from src.utils.config import apply_overrides


def test_grid_is_every_combination():
    trials = grid_trials({'lr': [0.1, 0.01], 'weight_decay': [0, 1, 2]})
    assert len(trials) == 6 and {'lr': 0.01, 'weight_decay': 2} in trials


def test_random_is_reproducible_and_in_range():
    space = {'lr': {'log_uniform': [1e-4, 1e-2]}, 'bs': [32, 64]}
    a, b = random_trials(space, 5, seed=3), random_trials(space, 5, seed=3)
    assert a == b and all(1e-4 <= t['lr'] <= 1e-2 and t['bs'] in (32, 64) for t in a)


def test_search_writes_comparison_and_best_params(tiny_cfg, fake_hf):
    cfg = apply_overrides(tiny_cfg, ['tuning.epochs=1', 'tuning.grid.lr=[0.001,0.003]',
                                     'tuning.random.trials=2', "tuning.random.space.lr={log_uniform: [0.0005, 0.005]}"])
    result = run_search(cfg, data={'train': fake_hf}, verbose=False)
    assert len(result['trials']) == 4
    assert set(result['comparison']['method']) == {'grid', 'random'}
    assert (result['output_dir'] / 'grid_vs_random.png').is_file()
    assert pd.read_csv(result['output_dir'] / 'grid_vs_random.csv')['trials'].tolist() == [2, 2]
    tuned = apply_tuned(cfg)
    assert tuned['training']['lr'] == result['best']['lr']
