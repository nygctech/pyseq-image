import os
import imageio
from os.path import join
from skimage.filters import gaussian

image_dir = '/Volumes/Kunal/HiSeqExperiments/20210323_4i4color/images'
test_dir = '/Users/kpandit/pyseq-image/src/demo/images'


test_sections = {'m1a':
                     {'height':128,
                     },
                 'm3b':
                     {'height':512,
                     }}
xpos = [11985, 12300]
opos = [25463, 25698, 25933]

channels = [558, 610, 687, 740]


for cy in range(1,6):
    for ch in channels:
        for x in xpos:
            im_name = f'c{ch}_A_sm4_r{cy}_x{x}_o25698.tiff'
            im = imageio.imread(join(image_dir,im_name))
            max_px = im.max()
            min_px = im.min()
            for t in test_sections.keys():
                height = test_sections[t]['height']
                cropped = im[3440:3440+height,:]
                for i, o in enumerate(opos):
                    if i % 2 == 0:
                        filt_im = gaussian(cropped.astype('float32'), sigma = 3).astype('uint16')
                    else:
                        filt_im = cropped
                    test_name = f'c{ch}_A_s{t}_r{cy}_x{x}_o{o}.tiff'
                    #print(test_name, filt_im.min(), filt_im.max())
                    imageio.imwrite(join(test_dir, test_name), filt_im)
