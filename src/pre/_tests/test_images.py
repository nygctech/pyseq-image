from pre import image_analysis as ia
import numpy as np
import pytest
from pathlib import Path

image_path = '/nethome/kpandit/NYGC-PySeq2500-Pipeline/src/demo/'

@pytest.fixture
def demo_image():
    image_path = Path(ia.__file__).parents[2] / Path('src/demo/images')
    # Reads 2 sections; m1a and m3b, return 1 it doesn't matter which
    ims = ia.get_HiSeqImage(image_path) 

    return ims[0]



def test_correct_background():

    raw_image = demo_image.im
    corrected_im = demo_image.correct_background_2()

    assert corrected_im.shape  == raw_image.shape
    assert np.max(corrected_im) == 4095
    assert np.min(corrected_im) >= 0
