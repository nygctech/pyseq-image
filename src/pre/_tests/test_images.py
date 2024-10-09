from pre import image_analysis as ia
import numpy as np
import pytest
from pathlib import Path
from math import ceil
import configparser
from tempfile import TemporaryDirectory



@pytest.fixture(params = ['m1a', 'm3b'])
def demo_image(request, demo_config):

    # TODO make test images from Origin and Varick and have
    # old config file for one and new yaml config for the other
    # also have demo configs in demo folder
    image_path = Path(ia.__file__).parents[2] / Path('src/demo/images')
    # Reads 2 sections; m1a and m3b, return 1 it doesn't matter which
    im = ia.HiSeqImages.open_tiffs(image_path, common_name=request.param, extra_config_path=demo_config)
    # im = ia.get_HiSeqImages(image_path, common_name = request.param, extra_config_path = demo_config)

    return im

@pytest.fixture()
def HSImage(demo_config):

    # TODO make test images from Origin and Varick and have
    # old config file for one and new yaml config for the other
    # also have demo configs in demo folder
    image_path = Path(ia.__file__).parents[2] / Path('src/demo/images')
    # Reads 2 sections; m1a and m3b, return 1 it doesn't matter which
    im = ia.HiSeqImages.open_tiffs(image_path, common_name='m1a', extra_config_path=demo_config)
    # im = ia.get_HiSeqImages(image_path, common_name = request.param, extra_config_path = demo_config)

    return im

@pytest.fixture(params = ['.cfg','.yaml'])
def demo_config(request):

    image_path = Path(ia.__file__).parents[1]
    config_path =(image_path / Path('demo/machine_settings')).with_suffix(request.param)

    return str(config_path)

def test_open_tiffs(demo_image):
    assert demo_image.config is not None
    assert demo_image.machine == 'Origin'
    assert demo_image is not None
    for d, d_ in zip(demo_image.im.dims, ['channel', 'cycle', 'obj_step', 'row', 'col']):
        assert d == d_


@pytest.fixture(scope = "session")
def test_write_ome_zarr(demo_image, tmp_path_factory):

    if not isinstance(demo_image.config, configparser.ConfigParser):
        omezarrpath = tmp_path_factory.mktemp('omezarr')
        assert demo_image.write_ome_zarr(omezarrpath)
        return omezarrpath
    
@pytest.fixture()
def xr_zarr(demo_image, tmp_path_factory):   
    
    # image_path = Path(ia.__file__).parents[2] / Path('src/demo/images')
    # config_path =  Path(ia.__file__).parents[2] / Path('src/demo/machine_settings.yaml')
    # im = ia.HiSeqImages.open_tiffs(image_path, common='m1a', extra_config_path=config_path)

    xrzarrpath = tmp_path_factory.mktemp('xrzarr')
    demo_image.save_zarr(xrzarrpath)
    return xrzarrpath
    
def test_xr_zarr(xr_zarr, demo_config):

    im = ia.HiSeqImages.open_zarr(xr_zarr, extra_config_path=demo_config)

    assert im is not None
    assert im.config is not None
    assert im.machine == 'Origin'
    for d, d_ in zip(im.im.dims, ['channel', 'cycle', 'obj_step', 'row', 'col']):
        assert d == d_


# parameterize for multiple images
def test_correct_background(demo_image):

    raw_image = demo_image.im
    corrected_im = demo_image.correct_background()
    assert corrected_im.shape  == raw_image.shape
    assert corrected_im.max().values == 4095
    assert corrected_im.min().values >= 0


def test_get_machine_config(demo_config):

    config, config_path = ia.get_machine_config('virtual', extra_config_path=demo_config)

    assert config is not None
    if config_path.suffix == '.yaml':
        assert len(config.get('background')) == 4
    elif config_path.suffix == '.cfg':
        assert len(config.options('virtualbackground')) == 4

def test_focus_projection(demo_image):


    #demo_image.correct_background()

    rows = demo_image.im.row.size
    cols = demo_image.im.col.size
    overlap = 0.5
    window = demo_image.im.chunksizes['col'][0]
    overlap = int(0.5*window)
    r = max(rows//overlap, 1)
    c = max(cols//overlap, 1)

    focus_map = demo_image.focus_projection()

    assert focus_map.shape == (r,c)
    assert demo_image.im.shape[-2:] == (rows, cols)
    assert np.all(np.logical_or(focus_map == 1, focus_map == 0))
    assert 'obj_step' not in demo_image.im.dims
