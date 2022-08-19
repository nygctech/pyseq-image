from pre import image_analysis as ia
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def demo_image():

    # TODO make test images from Origin and Varick and have
    # old config file for one and new yaml config for the other
    # also have demo configs in demo folder
    image_path = Path(ia.__file__).parents[2] / Path('src/demo/images')
    # Reads 2 sections; m1a and m3b, return 1 it doesn't matter which
    ims = ia.get_HiSeqImages(image_path)
    im = ims[0]

    return im

@pytest.fixture(params = ['.cfg','.yaml'])
def demo_config(request):

    image_path = Path(ia.__file__).parents[1]
    config_path = image_path / Path('demo/machine_settings' + request.param)

    return str(config_path)


# parameterize for multiple images
def test_correct_background(demo_image):

    raw_image = demo_image.im
    corrected_im = demo_image.correct_background()

    assert corrected_im.shape  == raw_image.shape
    assert corrected_im.max().values == 4095
    assert corrected_im.min().values >= 0


def test_get_machine_config(demo_config):

    config, config_path = ia.get_machine_config('virtual', demo_config)

    print(config_path)
    print(config)

    assert config is not None
    if config_path[-4:] == 'yaml':
        assert len(config.get('background')) == 4
    elif config_path[-3:] == 'cfg':
        assert len(config.options('virtualbackground')) == 4
