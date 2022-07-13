import pytest
from pathlib import Path


def experiment_config_paths():
    '''List of different types of demo experiment config paths.'''
    dir =  Path(__file__).parent
    cfg_path = dir/Path('demo_exp_config.cfg')
    cfg_str = str(cfg_path)

    return [cfg_str, cfg_path]

@pytest.fixture(scope='session', params=experiment_config_paths())
def exp_config_path(request):
    yield request.param
