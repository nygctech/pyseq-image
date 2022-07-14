import os
import imageio
from os.path import join

image_dir = '/Volumes/Kunal/HiSeqExperiments/20210323_4i4color/images'
test_dir = '/Users/kpandit/NYGC-PySeq2500-Pipeline/src/demo/images'


test_sections = ['m1a', 'm3b']


channels = [558, 610, 687, 740]

for cy in range(5):
    cy += 1
    for ch in channels:
        im_name = f'c{ch}_A_sm4_r{cy}_x12300_o25698.tiff'
        im = imageio.imread(join(image_dir,im_name))
        im = im[3440:3440+2048,:]

        for t in test_sections:
            test_name = f'c{ch}_A_s{t}_r{cy}_x12300_o25698.tiff'
            imageio.imwrite(join(test_dir, test_name), im)
