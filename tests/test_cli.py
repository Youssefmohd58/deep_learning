"""CLI wiring: every documented command parses and the offline check runs."""
import pytest

from src.cli import build_parser, main


@pytest.mark.parametrize('argv', [
    ['check'], ['eda'], ['augment-preview'], ['train', '--config', 'configs/resnet18.yaml', '--use-tuned'],
    ['tune', '--config', 'configs/tuning_resnet18.yaml', '--method', 'grid'], ['evaluate', '--model', 'all'],
    ['compare'], ['benchmark'], ['predict', '--checkpoint', 'x.pt', '--image', 'b.jpg', '--top-k', '3'],
    ['gradcam', '--checkpoint', 'x.pt', '--image', 'b.jpg', '--task', 'classification'],
])
def test_commands_parse(argv):
    assert build_parser().parse_args(argv).func


def test_unknown_task_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(['train', '--task', 'caption', '--config', 'configs/resnet18.yaml'])


def test_check_runs_offline(capsys):
    main(['check'])
    assert 'check PASSED' in capsys.readouterr().out
