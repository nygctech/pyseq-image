from pyseq import image_analysis as ia
import numpy as np


# FIX IMAGE PATH
image_path = '/Users/tommyly/image_path/src/demo/images'

def test_correct_background():

    image_path = '/Users/tommyly/image_path/src/demo/images'
    image = ia.get_HiSeqImages(image_path)
    image_2 = image[0]

    corrected_im = cb.correct_background_2(image[0])

    assert corrected_im.shape  == image_2.im.shape
    assert np.max(corrected_im) == 4095
    assert np.min(corrected_im) >= 0
