#!/usr/bin/python

import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
import zarr
import traceback
from collections import defaultdict



from math import log2, ceil, floor, log
from pathlib import Path
from os import path, getcwd, mkdir, makedirs
from scipy import stats
from scipy.spatial.distance import cdist
import imageio
import glob
import configparser
import time
import tabulate
#from qtpy.QtCore import QTimer
from skimage.registration import phase_cross_correlation
from skimage.filters import median as med_filter
from skimage.morphology import square
import yaml
from pre.utils import get_config, get_logger
import logging
import traceback 

from io import BytesIO
from dask import delayed
from dask_image.ndinterp import affine_transform


from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader
from ome_types.model import Instrument, Microscope, Objective, Image, Pixels, Channel, OME, TiffData
from ome_types import to_dict

xr.set_options(keep_attrs=True)

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
#from . import resources
#from .methods import userYN

# logger = logging.getLogger('ImageAnalysis')
logger = get_logger()


def message(logger, *args):
    """Print output text to logger or console.

       If there is no logger, text is printed to the console.
       If a logger is assigned, text is printed to the log file.

    """

    msg = 'ImageAnalysis::'
    for a in args:
        msg += str(a) + ' '

    if logger is None:
        print(msg)
    else:
        module.info(msg)


def sum_images(images, logger = None, **kwargs):
    """Sum pixel values over channel images.

       The image with the largest signal to noise ratio is used as the
       reference. Images without significant positive kurtosis, ie pixels that
       deviate from mean value (assumed as background), are discarded. The
       remaining image histograms are matched to the reference. Finally, the
       images are summed together. The summed image is returned or if there is
       no signal in all channels, False is returned.

       Parameters:
       - images (data array): Xarray data array of images
       - logger (logger): Logger object to record process.

       Return:
       - array: Xarray data array of summed image or None if no signal

    """

    name_ = 'SumImages:'
    sum_im = None
    thresh = [1, 9, 27, 81]

    mean_ = kwargs.get('mean', None)
    std_ = kwargs.get('std', None)
    # Calculate modified kurtosis
    channels = images.channel.values
    k_dict = {}
    for i, ch in enumerate(channels):
        if mean_ is not None:
            mean = mean_[i]
        else:
            mean = None
        if std_ is not None:
            std = std_[i]
        else:
            std = None

        k = kurt(images.sel(channel=ch), mean=mean, std=std)
        message(logger, name_, 'Channel',ch, 'k = ', k)
        print(name_, 'Channel',ch, 'k = ', k)
        k_dict[ch] = k

    # Pick kurtosis threshold
    max_k = max(list(k_dict.values()))
    thresh_ind = np.where(max_k>np.array(thresh))[0]
    if len(thresh_ind) > 0:
        thresh = thresh[max(thresh_ind)]
        message(logger, name_, 'kurtosis threshold (k) = ', thresh)

        # keep channels with high kurtosis
        keep_ch = [ch for ch in channels if k_dict[ch] > thresh]
        im = images.sel(channel = keep_ch)

        # Sum remaining channels
        im = im.sum(dim='channel')
    else:
        im = None

    return im

def interpolate(image, new_min=0, new_max=1, old_min=None, old_max=None, dtype = None):
    '''Interpolate image from old min/max to new min/max.'''

    if dtype is None:
        dtype = image.dtype
    else:
        assert dtype in ['uint8']

    if old_min is None:
        old_min = image.min()

    if old_max is None:
        old_max = image.max()


    image = (image-old_min) * (new_max - new_min) / (old_max-old_min) + new_min

    return image.astype(dtype)


def kurt(im, mean=None, std=None):
    """Return kurtosis = mean((image-mode)/2)^4). """
    print(mean, std)
    if mean is None:
        mean = stats.mode(im, axis = None)[0][0]
    if std is None:
        std = 2
    z_score = (im-mean)/std
    z_score = z_score**4
    k = float(z_score.mean().values)

    return k

def get_focus_points(im, scale, min_n_markers, log=None, p_sat = 99.9):
    """Get potential points to focus on.

       First 1000 of the brightest, unsaturated pixels are found.
       Then the focus field of views with the top *min_n_markers* contrast are
       ordered based on the distance from each other, with the points farthest
       away from each other being first.

       **Parameters:**
       - im (array): Summed image across all channels with signal.
       - scale (int): Factor at which the image is scaled down.
       - min_n_markers (int): Minimun number of points desired, max is 1000.
       - p_sat (float): Percentile to call pixels saturated.
       - log (logger): Logger object to record process.


       **Returns:**
       - array: Row, Column list of ordered pixels to use as focus points.

    """

    name_ = 'GetFocusPoints::'
    #score pixels
    im = im.values
    px_rows, px_cols = im.shape
    px_sat = np.percentile(im, p_sat)
    px_score = np.reshape(stats.zscore(im, axis = None), (px_rows, px_cols))

    # Find brightest unsaturated pixels
    edge_width = int(2048/scale/2)
    im_ = np.zeros_like(im)
    px_score_thresh = 3
    while np.sum(im_ != 0) < min_n_markers:
        # Get brightest pixels
        im_[px_score > px_score_thresh] = im[px_score > px_score_thresh]
        # Remove "saturated" pixels
        im_[im > px_sat] = 0
        #Remove Edges
        if edge_width < px_cols/2:
          im_[:, px_cols-edge_width:px_cols] = 0
          im_[:,0:edge_width] = 0
        if edge_width < px_rows/2:
          im_[0:edge_width,:] = 0
          im_[px_rows-edge_width:px_rows, :] = 0

        px_score_thresh -= 0.5


    px_score_thresh += 0.5
    message(log, name_, 'Used', px_score_thresh, 'pixel score threshold')
    markers = np.argwhere(im_ != 0)


    # Subset to 1000 points
    n_markers = len(markers)
    message(log, name_, 'Found', n_markers, 'markers')
    if n_markers > 1000:
      rand_markers = np.random.choice(range(n_markers), size = 1000)
      markers = markers[rand_markers,:]
      n_markers = 1000

    # Compute contrast
    c_score = np.zeros_like(markers[:,1])
    for row in range(n_markers):
      mark = markers[row,:]
      if edge_width < px_cols/2:
        frame = im[mark[0],mark[1]-edge_width:mark[1]+edge_width]
      else:
        frame = im[mark[0],:]
      c_score[row] = np.max(frame) - np.min(frame)


    # Get the minimum number of markers needed with the highest contrast
    if n_markers > min_n_markers:
      p_top = (1 - min_n_markers/n_markers)*100
    else:
      p_top = 0
    message(log, name_, 'Used', p_top, 'percentile cutoff')
    c_cutoff = np.percentile(c_score, p_top)
    c_markers = markers[c_score >= c_cutoff,:]

    # Compute distance matrix
    dist = cdist(c_markers, c_markers)
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple

    # Order marker points based on distance from each other
    # Markers farthest from each other are first
    n_markers = len(c_markers)
    ord_points = np.zeros_like(c_markers)
    if n_markers > 2:
        ord_points[0,:] = c_markers[max_ind[0],:]
        ord_points[1,:] = c_markers[max_ind[1],:]
        _markers = np.copy(c_markers)
        prev2 = max_ind[0]
        prev1 = max_ind[1]
        dist = np.delete(dist,[prev2,prev1],1)
        _markers = np.delete(_markers,[prev2,prev1], axis=0)
        for i in range(2,n_markers):
          dist2 = np.array([dist[prev2,:],dist[prev1,:]])
          ind = np.argmax(np.sum(dist2,axis=0))
          ord_points[i,:] = _markers[ind,:]
          dist = np.delete(dist,ind,1)
          _markers = np.delete(_markers,ind, axis=0)
          prev2 = prev1
          prev1 = ind
    else:
        ord_points = c_markers

    return ord_points

def get_focus_points_partial(im, scale, min_n_markers, log=None, p_sat = 99.9):
    """Get potential points to focus on.

       First 1000 of the brightest, unsaturated pixels are found.
       Then the focus field of views with the top *min_n_markers* contrast are
       ordered based on the distance from each other, with the points farthest
       away from each other being first.

       **Parameters:**
       - im (array): Summed image across all channels with signal.
       - scale (int): Factor at which the image is scaled down.
       - min_n_markers (int): Minimun number of points desired, max is 1000.
       - p_sat (float): Percentile to call pixels saturated.
       - log (logger): Logger object to record process.


       **Returns:**
       - array: Row, Column list of ordered pixels to use as focus points.

    """

    name_ = 'GetFocusPointsPartial::'
    im = im.values
    #score pixels
    px_rows, px_cols = im.shape
    px_sat = np.percentile(im, p_sat)
    px_score = np.reshape(stats.zscore(im, axis = None), (px_rows, px_cols))

    # Find brightest unsaturated pixels
    edge_width = int(2048/scale/2)
    im_ = np.zeros_like(im)
    px_score_thresh = 3
    while np.sum(im_ != 0) < min_n_markers:
        # Get brightest pixels
        im_[px_score > px_score_thresh] = im[px_score > px_score_thresh]
        # Remove "saturated" pixels
        im_[im > px_sat] = 0
        #Remove Edges
        if edge_width < px_cols/2:
          im_[:, px_cols-edge_width:px_cols] = 0
          im_[:,0:edge_width] = 0
        if edge_width < px_rows/2:
          im_[0:edge_width,:] = 0
          im_[px_rows-edge_width:px_rows, :] = 0

        px_score_thresh -= 0.5


    px_score_thresh += 0.5
    message(log, name_, 'Used', px_score_thresh, 'pixel score threshold')
    markers = np.argwhere(im_ != 0)

    #Subset to unique y positions
    markers = np.array([*set(markers[:,0])])

    # Subset to 1000 points
    n_markers = len(markers)
    message(log, name_, 'Found', n_markers, 'markers')
    if n_markers > 1000:
      rand_markers = np.random.choice(range(n_markers), size = 1000)
      markers = markers[rand_markers]
      n_markers = 1000



    # Compute contrast
    c_score = np.zeros_like(markers)
    for row in range(n_markers):
      frame = im[markers[row]]
      c_score[row] = np.max(frame) - np.min(frame)

    # Get the minimum number of markers needed with the highest contrast
    if n_markers > min_n_markers:
      p_top = (1 - min_n_markers/n_markers)*100
    else:
      p_top = 0
    message(log, name_, 'Used', p_top, 'percentile cutoff')
    c_cutoff = np.percentile(c_score, p_top)
    c_markers = markers[c_score >= c_cutoff]

    n_markers = len(c_markers)
    dist = np.ones((n_markers,n_markers))*c_markers
    dist = abs(dist-dist.T)
    # Compute distance matrix
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple

    # Order marker points based on distance from each other
    # Markers farthest from each other are first
    ord_points = np.zeros_like(c_markers)
    if n_markers > 2:
        ord_points[0] = c_markers[max_ind[0]]
        ord_points[1] = c_markers[max_ind[1]]
        _markers = np.copy(c_markers)
        prev2 = max_ind[0]
        prev1 = max_ind[1]
        dist = np.delete(dist,[prev2,prev1], 1)
        _markers = np.delete(_markers,[prev2,prev1])
        for i in range(2,n_markers):
          dist2 = np.array([dist[prev2,:],dist[prev1,:]])
          ind = np.argmax(np.sum(dist2,axis=0))
          ord_points[i] = _markers[ind]
          dist = np.delete(dist,ind,1)
          _markers = np.delete(_markers,ind)
          prev2 = prev1
          prev1 = ind
    else:
        ord_points = c_markers

    ord_points = np.array([c_markers,np.ones(n_markers)*edge_width]).T

    return ord_points


def compute_background(image_path=None, common_name = ''):


    im = get_HiSeqImages(image_path, common_name)
    config, config_path = get_machine_config(im.machine)
    config_section = im.machine+'background'
    try:
        im = im[0] # In case there are multiple sections in image_path
    except:
        pass

    if im.machine == 'virtual':
        sensor_size = 32
    else:
        sensor_size = 256 # pixels

    # Check if background data exists and check with user to overwrite
    bg_dict = True
    if config.has_section(config_section):
        print('Current background correction')
        print(tabulate.tabulate(config.items(config_section),
              tablefmt='presto',headers=['channel','background correction']))
        if not userYN('Calculate new background correction for '+im.machine):
            bg_dict = None

    if bg_dict:
        print('Analyzing ', im.im.name)
        bg_dict = {}
        # Loop over channels then sensor group and find mode of sensor group
        for ch in im.im.channel.values:
            background = []
            for i in range(8):
                sensor = im.im.sel(channel=ch, col=slice(i*sensor_size,(i+1)*sensor_size))
                background.append(stats.mode(sensor, axis=None)[0][0])
            avg_background = int(round(np.mean(background)))
            print('Channel', ch,'::Average background', avg_background)

            for i in range(8):
                background[i] = avg_background-background[i]                    # Calculate background correction
            print('Channel', ch,'::',*background)
            bg_dict[ch] = ','.join(map(str, background))                        # Format backround correction

        if userYN('Save new background data for '+im.machine):
            # Save background correction values in config file
            config.read_dict({config_section:bg_dict})
            with open(config_path,'w') as f:
                    config.write(f)

    return bg_dict




def get_HiSeqImages(image_path=None, common_name='', **kwargs):

    # Get and sort image files
    if image_path is None:
        image_path = getcwd()
    image_path = Path(image_path)
    files = get_image_files(image_path, ['.zarr', '.tiff'], common=common_name)
    if isinstance(files, dict):
        zarr_files = files['.zarr']
        tiff_files = files['.tiff']
    elif isinstance(files, list):
        if files[0].suffix == '.tiff':
            tiff_files = files
            zarr_files = []
        if files[0].suffix == '.zarr':
            zarr_files = files
            tiff_files = []

    # Open images
    images = []
    if len(zarr_files) > 0:
        for f in zarr_files:
            images.append(HiSeqImages.open_zarr(f, **kwargs))
    if len(tiff_files) > 0:
        ims = HiSeqImages.open_tiffs(files=files, **kwargs)
        if isinstance(ims, list):
            images += ims 
        elif ims is not None:
            images.append(ims)

    # Return Images
    n_images = len(images)
    if n_images > 1:
        return images
    elif n_images == 1:
        return images[0]
    else:
        raise FileNotFoundError(f'No images found in {image_path}')
    

    # files = get_image_files(path, 'zarr', common)
    #     images = []
    #     for f in files:
    #         try:             
    #             ome_metadata = zarr.open_group(f, mode = 'r').attrs["omero"]
    #             im = cls.open_ome_zarr(ome_metadata, path, **kwargs)
    #         except KeyError:
    #             im = cls.open_xr_zarr(path, **kwargs)      

    #         if isinstance(im, list):
    #             images =+ im 
    #         else:
    #             images.append(im)

    #     n_images = len(images)    
    #     if n_images > 1:
    #         return images
    #     elif n_images == 1:
    #         return images[0]


def get_machine_config(machine, extra_config_path=None, **kwargs):
    '''Get machine settings config from default location.

      Default locations in order of preference:
      - ~/.config/pyseq2500/machine_settings.yaml
      - ~/.config/pyseq2500/machine_settings.cfg
      - ~/.pyseq2500/machine_settings.yaml
      - ~/.pyseq2500/machine_settings.cfg

      Parameters:
      machine (str): Name of machine
      extra_config_path (str): Path to machine setting if not in defaul location

      Returns:
      config: Machine settings config (ConfigParser or YAML)
      config_path: Machine settings config path (str)
    '''

    config_paths = [Path.home() / '.config/pyseq2500/machine_settings',
                    Path.home() / '.pyseq2500/machine_settings']
    if extra_config_path is not None:
        config_paths.insert(0, Path(extra_config_path))

    config_path = None
    for p in config_paths:
        if p.with_suffix('.yaml').exists():
            config_path = p.with_suffix('.yaml')
            break
        elif p.with_suffix('.cfg').exists():
            config_path = p.with_suffix('.cfg')
            break

    if config_path is None:
        logger.error(f'Machine settings config not found')
        config = None
    else:
        config = get_config(config_path)

        if config_path.suffix == '.yaml':
            config = config.get(machine, None)
        elif config_path.suffix == '.cfg':
            machine = str(machine).lower()
            if machine not in config.options('machines'):
                config = None

    if config is None:
        logger.error(f'Settings for {machine} do not exist')

    return config, config_path


def detect_channel_shift(image_path, common_name = '', ref_ch = 610):

    im = get_HiSeqImages(image_path, common_name)
    im = im.im
    config, config_path = get_machine_config(im.machine)

    try:
        im = im[0]                                                              # In case there are multiple sections in image_path
    except:
        pass

    if 'obj_step' in im.dims:
        im = im.max(dim='obj_step')
    if 'cycle' in im.dims:
        if len(im.cycle.values) > 1:
            im = im.sel(cycle=1)
    if 'image' in im.dims:
        if len(im.image.values) > 1:
            im = im.sel(image=1)

    channels = list(im.channel.values)
    if len(channels) != 4:
        raise ValueError('Need 4 channels.')

    if ref_ch in channels:
        channels.pop(channels.index(ref_ch))
    else:
        print('Invalid reference channel')
        raise ValueError

    # detect shift
    shift = []
    for ch in channels:
        detected_shift = phase_cross_correlation(im.sel(channel=ref_ch),im.sel(channel=ch))
        row_shift = int(detected_shift[0][0])
        col_shift = int(detected_shift[0][1])

        print(ch, 'row shift =', row_shift, 'px')
        print(ch, 'col shift =', col_shift, 'px')

        if row_shift > 0:
            shift += [0, -row_shift]
        elif row_shift < 0:
            shift += [-row_shift, 0]
        else:
            shift += [0, 0]
        if col_shift > 0:
            shift += [0, -col_shift]
        elif col_shift < 0:
            shift += [-col_shift, 0]
        else:
            shift += [0, 0]

    shift = np.reshape(shift, (nch-1, 4))

    # adjust for global pixel shifts
    max_row_top = np.max(shift[:,0])
    min_row_bot = abs(np.min(shift[:,1]))
    max_col_l = np.max(shift[:,2])
    min_col_r = abs(np.min(shift[:,3]))


    ch_shift = {}
    ch_list = []
    for i, ch in enumerate(channels):
        # adjust row shift
        if shift[i,0] != 0 and shift[i,1] == 0:
            shift[i,0] = shift[i,0] + min_row_bot
        elif shift[i,1] != 0 and shift[i,0] == 0:
            shift[i,1] = shift[i,1] - max_row_top

        #adjust col shift (hopefully none)
        if shift[i,2] != 0 and shift[i,3] == 0:
            shift[i,2] = shift[i,2] + min_col_r
        elif shift[i,3] != 0 and shift[i,2] == 0:
            shift[i,3] = shift[i,3] - max_col_l

        #Replace 0 with None
        ch_shift[ch] = [None if s == 0 else s for s in shift[i,:]]
        #Shift image
        s = ch_shift[ch]
        ch_list.append(im.sel(channel=ch, row=slice(s[0],s[1]), col=slice(s[2],s[3])))

    # Shift reference channel
    ch_shift[ref_ch] = [min_row_bot, -max_row_top, min_col_r, max_col_l]
    ch_shift[ref_ch] = [None if s == 0 else s for s in ch_shift[ref_ch]]
    s = ch_shift[ref_ch]
    ch_list.append(im.sel(channel=ref_ch, row=slice(s[0],s[1]), col=slice(s[2],s[3])))


    # Show resulting shift with original image
    shifted = xr.concat(ch_list, dim='channel')
    both = xr.concat([im.sel(row=slice(s[0],s[1]),col=slice(s[2],s[3])), shifted], dim='shifted')
    both = ia.HiSeqImages(im = both)
    both.show()

    # save shift
    if userYN('Save channel shift to machine settings'):
        reg_dict = {}
        for ch in ch_shift.keys():
            reg_dict[str(ch)] = ','.join(map(str, ch_shift[ch]))

        #config, config_path = get_machine_config(im.machine)
        config.read_dict({im.machine+'registration':reg_dict})

        if path.exists(config_path):
            with open(config_path,'w') as config_path_:
                config.write(config_path_)

    return ch_shift

def get_image_files(path, suffix, common=''):

    if isinstance(suffix, str):
        suffix = [suffix]
    for i, s in enumerate(suffix):
        if s[0] != '.':
            suffix[i] = f'.{s}'

    path = Path(path)
    files = defaultdict(list)

    if path.suffix == '.zarr' and common in path.stem:
        files[path.suffix].append(path)

    if path.is_dir():
        for f in path.iterdir():
            if f.suffix in suffix and common in f.stem:
                files[f.suffix].append(f)

    if len(files) == 0:
        raise FileNotFoundError(f'No files with {suffix} found in {path}')
    elif len(files) == 1:
        return files[list(files.keys())[0]]
    else:
        return files




class HiSeqImages():
    """HiSeqImages

       **Attributes:**
        - im (dict): Xarray DataArray N-Dimensional Image
        - channel_color (dict): Dictionary of colors to display each channel as
        - files: Files used to stitch image


    """
    machine = None
    resolution = 0.375                                                 # um/px
    x_spum = 0.4096                                                    #steps per um
    channel_color = {558:'blue', 610:'green', 687:'magenta', 740:'red'}

    def __init__(self, im, machine=None, files=[], **kwargs):
        
        # image_path=None, common_name='',  im=None,
        #                obj_stack=None, RoughScan = False, **kwargs):
        """The constructor for HiSeq Image Datasets.

           **Parameters:**
            - image_path (path): Path to images with tiffs

           **Returns:**
            - HiSeqImages object: Object to manipulate and view HiSeq image data

        """

        self.im = im
        self.name = im.name
        self.files = files
        if machine is None:
            self.assign_machine()
        else:
            self.machine = machine
        if self.machine is not None:
            self.config, config_path = get_machine_config(im.machine, **kwargs)
        

        # if im is None:

        #     if image_path is None:
        #         image_path = getcwd()
        #     else:
        #         image_path = str(image_path)

        #     # Get machine config
        #     name_path = path.join(image_path,'machine_name.txt')
        #     if path.exists(name_path):
        #         with open(name_path,'r') as f:
        #             machine = f.readline().strip()
        #         self.config, config_path = get_machine_config(machine, **kwargs)
        #     if self.config is not None:
        #         self.machine = machine
        #     if self.machine is None:
        #         self.machine = ''

        #     if len(common_name) > 0:
        #         common_name = '*'+common_name

        #     section_names = []

        #     # Open zarr
        #     if image_path[-4:] == 'zarr':
        #         self.filenames = [image_path]
        #         section_names = self.open_zarr()

        #     elif obj_stack is not None:
        #         # Open obj stack (jpegs)
        #         #filenames = glob.glob(path.join(image_path, common_name+'*.jpeg'))
        #         n_frames = self.open_objstack(obj_stack)

        #     elif RoughScan:
        #         # RoughScans
        #         filenames = glob.glob(path.join(image_path,'*RoughScan*.tiff'))
        #         if len(filenames) > 0:
        #             self.filenames = filenames
        #             n_tiles = self.open_RoughScan()

        #     else:
        #         # Open tiffs
        #         filenames = glob.glob(path.join(image_path, common_name+'*.tiff'))
        #         if len(filenames) > 0:
        #             self.filenames = filenames
        #             section_names += self.open_tiffs()

        #         # Open zarrs
        #         filenames = glob.glob(path.join(image_path, common_name+'*.zarr'))
        #         if len(filenames) > 0:
        #             self.filenames = filenames
        #             try:             
        #                 ome_metadata = zarr.open_group(path, mode = 'r').attrs["omero"]
        #                 sections_names += self.read_ome_zarr(ome_metadata)
        #             except:
        #                 section_names += self.open_zarr()

        #     if len(section_names) == 1:
        #         self.name = section_names[0]

        #     if len(section_names) > 0:
        #         section_names = ' '.join(section_names)
        #         self.logger.info(f'Opened {section_names}')


        # else:
        #     self.machine = im.machine
        #     if im.machine is not None:
        #         self.config, config_path = get_machine_config(im.machine)
        #     self.im = im
        #     self.name = im.name

    def assign_machine(self):
        config_path = Path.home() / '.config/pyseq2500/machine_settings.yaml'
        if isinstance(self.files, list):
            name_path = self.files[0].parent/'machine_name.txt'
            if name_path.exists():
                with open(name_path,'r') as f:
                    self.machine = f.readline().strip()
        elif config_path.exists():
            config = get_config(config_path)
            self.machine = config.get('name', None)
        else:
            logger.warning('Could not assign machine')

        self.im.attrs['machine'] = self.machine


    def correct_background(self):
        # Maintain compatibility with older config files and
        # background correction parameters meant for subtraction
        if isinstance(self.config, configparser.ConfigParser):
            im = self.correct_background_subtract()
        # YAML config with background correction parameters meant for rescaling
        elif isinstance(self.config, dict):
            im = self.correct_background_rescale()
        else:
            logger.warning('CorrectBackground::Invalid config')
            im = None

        return im


    def correct_background_subtract(self):
        '''Subtract background from all groups to average background value.'''

        machine = self.machine
        if not bool(self.im.fixed_bg) and machine is not None:
            bg_dict = {}
            for ch in self.im.channel.values:
                values  = self.config.get(str(machine)+'background', str(ch))
                values = list(map(int,values.split(',')))
                bg_dict[int(ch)]=values

            ch_list = []
            ncols = len(self.im.col)
            for ch in self.im.channel.values:
                bg_ = np.zeros(ncols)
                i = 0
                for c in range(int(ncols/256)):
                    if c == 0:
                        i = self.im.first_group
                    if i == 8:
                        i = 0
                    bg = bg_dict[ch][i]
                    bg_[c*256:(c+1)*256] = bg
                    i += 1
                ch_list.append(self.im.sel(channel=ch)+bg_)
            self.im = xr.concat(ch_list,dim='channel')
            self.im.attrs['fixed_bg'] = 1

        else:
            pre_msg='CorrectBackground::'
            if bool(self.im.fixed_bg):
                logger.info(pre_msg+'Image already background corrected.')
            elif machine is None:
                logger.warning(pre_msg+'Unknown machine')

        return self.im

    def correct_background_rescale(self):
        '''Rescale pixel values for each group to the average background.'''

        new_min_dict = self.config.get('background')
        dark_dict = self.config.get('dark group')
        max_px = self.config.get('max_pixel_value')

        pre_msg = 'CorrectBackgroundRescale'
        logger.debug(f'{pre_msg} :: max px :: {max_px}')

        ncols = len(self.im.col)
        ntiles = int(ncols/2048)
        gs = 256 if max_px == 4095 else 512
        max_px_ = da.from_array([max_px] * ncols)
        ngroups = int(2048/gs)

        ch_list = []
        for ch in self.im.channel.values:
            new_min = new_min_dict[ch]
            new_min_ = da.from_array([new_min] * ncols)
            logger.debug(f'{pre_msg} :: channel {ch} min px :: {new_min}')

            group_min_ = np.zeros(ncols)
            for t in range(ntiles):
                for g in range(ngroups):
                    group_min_[t*2048 + g*gs : t*2048 + (g+1)*gs] = dark_dict[ch][g]
            group_min_ = da.array(group_min_)

            old_contrast = max_px_ - group_min_
            new_contrast = da.from_array([max_px - new_min] * ncols)
            plane = self.im.sel(channel=ch)
            corrected = (((plane-group_min_).clip(min=0)/old_contrast * new_contrast) +  new_min_).astype('uint16')
            ch_list.append(corrected)

        print(self.im)
        _dims = tuple(['channel']+list(corrected.dims))
        logger.error(_dims)

        self.im = xr.concat(ch_list, dim='channel')
        print(self.im)
        self.im.name = self.name

        return self.im

    def register_channels(self):
        # Maintain compatibility with older config files and
        # background correction parameters meant for subtraction
        if isinstance(self.config, configparser.ConfigParser):
            im = self.register_channels_shift()
        # YAML config with background correction parameters meant for rescaling
        elif isinstance(self.config, dict):
            im = self.register_channels_affine()
        else:
            logger.warning('Invalid config')
            im = None

        return im

    def register_channels_shift(self, image=None):
        """Register image channels."""

        if image is None:
            img = self.im
        else:
            img = image
        machine = self.machine
        config_section = str(machine)+'registration'

        reg_dict = {}
        if self.config.has_section(config_section):
            # Format values from config
            for ch, values in self.config.items(config_section):
                pxs = []
                for v in values.split(','):
                    try:
                        pxs.append(int(v))
                    except:
                        pxs.append(None)
                reg_dict[int(ch)]=pxs


            shifted=[]
            for ch in reg_dict.keys():
                shift = reg_dict[ch]
                shifted.append(img.sel(channel = ch,
                                       row=slice(shift[0],shift[1]),
                                       col=slice(shift[2],shift[3])
                                       ))

            img = xr.concat(shifted, dim = 'channel')
            img = img.sel(row=slice(64,None))                                   # Top 64 rows have white noise

            if image is None:
                self.im = img
        else:
            logger.warning('registerChannelsShift :: Unknown machine')

        return img

    def update_crop_bb(self, crop_bb, shift):

        if shift[0] < 0:
            crop_bb[1] = max(crop_bb[1], floor(-shift[0]))
        else:
            crop_bb[0] = max(crop_bb[0], ceil(shift[0]))

        if shift[1] < 0:
            crop_bb[2] = max(crop_bb[2], floor(-shift[1]))
        else:
            crop_bb[3] = max(crop_bb[3], ceil(shift[1]))

        return crop_bb

    def get_registration_data(self, image=None, top=64, bottom=0, left=0, right=0):
        """Get registration shift for each channel from config and crop bounding box"""

        img = self.im

        # Get registration data
        reg_config = self.config.get('registration', None)

        pre_msg = 'getRegistrationData ::'
        logger.info(f'{pre_msg} {self.machine} registration data')

        reg_dict = {}
        crop_bb = [top, bottom, left, right]

        if reg_config is not None:
            for ch, shift in reg_config.items():

                assert len(shift) == 2
                try:
                    shift = [float(s) for s in shift]
                except:
                    logger.warning(f'{pre_msg} Registration shift {shift} for channel {ch} is invalid')

                A = np.identity(3)
                A[0,2] = -shift[0]
                A[1,2] = -shift[1]
                reg_dict[int(ch)]=A
                crop_bb = self.update_crop_bb(crop_bb, shift)

                logger.info(f'{pre_msg} Channel {ch} :: {shift}')

            logger.info(f'{pre_msg} :: Crop bounding box :: {crop_bb} (top, bottom, left, right)')
        else:
            raise ValueError(f'Registration data for {machine} not found')

        return reg_dict, crop_bb



    def apply_full(self, func, dim_depth = None, sel_dict = None, dim_stack = None, args = (), kwargs = {}):
        '''Recursively loop over coordinates and apply function to 2D images.

           Function should take 2D image as it's first argument and return only
           a 2D xarray image with fully labeled coordinates.

           Returns:
           ND xr.DataArray with function applied to entire DataArray
        '''

        assert callable(func)

        # assert that selecting all coords results in 2D image

        dims = [d for d in self.im.dims if d in self.im.coords.keys()]
        max_dim_depth = len(dims)

        if dim_depth is None:
            dim_depth = 0
        if sel_dict is None:
            sel_dict = dict()
        if dim_stack is None:
            dim_stack = dict.fromkeys(dims, [])


        dim = dims[dim_depth]
        dim_stack[dim] = []

        # Loop over coords recursively
        for value in self.im.coords[dim]:
            sel_dict[dim] = value

            if dim_depth < max_dim_depth-1:
                self.apply_full(func, dim_depth + 1, sel_dict, dim_stack, args, kwargs)
            else:
                # apply function to
                image = self.im.sel(sel_dict)
                assert len(image.shape) == 2, 'More than 2 dimensions do not have coordinates'
                image = func(image, *args, **kwargs)

            # save plane in stack
            if dim == dims[-1]:
                dim_stack[dim].append(image)

        # stack dimensions
        if dim != dims[0]:
            higher_dim = dims[dim_depth-1]
            dim_stack[higher_dim].append(xr.concat(dim_stack[dim], dim))
        else:
            return xr.concat(dim_stack[dim], dim)




    def register_and_crop(self, image, reg_dict, crop_bb):
        '''Register images with affine transformation and crop with bounding box.

            crop_bb = [top bottom left right] pixel bounding box

            Parameters:
            image: 2D image
            reg_dict: Channels as keys and affine matrix as values
            crop_bb: 1D array crop bounding box in pixel location

            Returns:
            2D image
        '''

        rows = len(image.row); cols = len(image.col)
        ch = int(image.channel.values)

        # Affine transformation
        if ch in reg_dict.keys():
            registered =  affine_transform(image.data, reg_dict[ch])
            image = xr.DataArray(registered, name = image.name, dims = image.dims,
                                 coords = image.coords, attrs = image.attrs)

        cols_ = slice(crop_bb[2], cols-crop_bb[3])
        rows_ = slice(crop_bb[0], rows-crop_bb[1])
        if crop_bb[2] > 0 and crop_bb[3] == 0:
            # Flip columns if left side of image cropped so irregular chunk is last
            image = image.sel(col=slice(None,None,-1))
            cols_ = slice(0, cols-crop_bb[2])
        elif crop_bb[2] > 0 and crop_bb[3] > 0:
            # Don't crop both left and right side, only right so only 1 irregular chunk and is last
            cols_ = slice(0, cols-crop_bb[3])

        # Crop image
        return image.sel(row=rows_, col=cols_)



    def register_channels_affine(self):

        reg_dict, crop_bb = self.get_registration_data()
        self.im = self.apply_full(self.register_and_crop, args = (reg_dict, crop_bb))

        return self.im

    def focus_projection(self, overlap = 0.5, cycle=1, channel=610, smooth = True):
        '''Project best focus Z slice across XY window and reduce 3D to 2D.'''

        nrows = self.im.row.size
        ncols = self.im.col.size
        nobj_steps = self.im.obj_step.size
        
        logger.info(f'Projecting cycle {cycle}, channel {channel}')
        image = self.im.sel(cycle=1, channel=610)
        im_min = image.min().values
        im_max = image.max().values

        col_chunk = image.chunksizes['col'][0]
        row_chunk = image.chunksizes['row'][0]
        window = col_chunk
        logger.info(f'Window size = {window} pixels')

        filter_size = ceil(1/overlap) + 1
        overlap = int(overlap*window)
        _rows = max(nrows//overlap,1); _cols = max(ncols//overlap,1)


        # Find size of sliding window image as a jpeg
        @delayed
        def _get_jpeg(tile):

            jpeg_size = np.zeros((_rows,1), 'uint8')
            tile = tile.values

            for r in range(_rows):
                rows = slice(r*overlap, min(nrows, (r*overlap)+window))
                fov = tile[rows,:]
                if im_max > 255 or im_min < 0:
                    fov = interpolate(fov, new_min = 0, new_max=255, old_min = im_min, old_max = im_max, dtype='uint8')
                with BytesIO() as f:
                    imageio.imwrite(f, fov, format='jpeg')
                    jpeg_size[r] = f.__sizeof__()

            return jpeg_size

        # Measure focus of window
        o_stack = []
        for o_ind, o in enumerate(image.obj_step):
            col_stack = []
            for c in range(_cols):
                cols = slice( c*overlap, min(ncols, (c*overlap)+window))
                tile = image.sel(col = cols, obj_step = o)
                tile_focus_vals = da.from_delayed(_get_jpeg(tile), shape = (_rows,1), dtype = 'uint8' )
                col_stack.append(tile_focus_vals)
            o_stack.append(da.hstack(col_stack))
        focus_vals = da.stack(o_stack, axis=2)


        # Find z slice that is most in focus for each window
        focus_map = focus_vals.argmax(axis = 2)
        logger.info(f'Begin computing focus map from {focus_map.size} windows')
        focus_map = focus_map.compute()
        logger.info('Finished computing focus map')
        logger.debug('Focus Map')
        logger.debug(focus_map)

        # Median filter focus map
        if smooth:
            focus_map = med_filter(focus_map, square(filter_size)).astype('uint8')


        # Build 2D image from most in focus frames
        obj_steps = self.im.obj_step
        col_stack = []
        for c in range(_cols):
            c_end = (c+1)*overlap if c < _cols-1 else ncols
            cols = slice(c*overlap, c_end)
            row_stack = []
            for r in range(_rows):
                r_end = (r+1)*overlap if r < _rows-1 else nrows
                rows = slice(r*overlap, r_end)
                o_ind = focus_map[r, c]
                row_stack.append(self.im.sel(row=rows, col = cols, obj_step=obj_steps[o_ind]))
            col_stack.append(xr.concat(row_stack, dim = 'row'))
        focus_image = xr.concat(col_stack, dim = 'col')
        
        # Rechunk
        focus_image = focus_image.chunk({'row':row_chunk, 'col':col_chunk})
        self.im = focus_image

        return focus_map

    def normalize(self, dims=['channel']):
        '''Normalize pixel values between 0 and 1.

           Parameters:
           - keep_dims ([str]): list of dims to normalize over

        '''


        # Check keep_dims exist and find dims to compute min/max over
        for d in images.dims:
            assert d in images.dims, f'{d} not in dims'
        dims_ = [d for d in images.dims if d != dims]

        min_px = images.min(dim=dims_)
        max_px = images.max(dim=dims_)
        channels = images.channel.values

        for ch in enumerate(channels):
            _min_px = min_px.sel(channel=ch).values
            _max_px = max_px.sel(channel=ch).values
            message(logger, name_, 'Channel',ch, 'min px =', _min_px, 'max px =', _max_px)

        self.im = (images-min_px)/(max_px-min_px)


    def remove_overlap(self, overlap=0, direction = 'left'):
        """Remove pixel overlap between tile."""

        try:
            overlap=int(overlap)
            n_tiles = ceil(len(self.im.col)/2048)
        except:
            logger.error('overlap must be an integer')

        try:
            if direction.lower() in ['l','le','lef','left','lft','lt']:
                direction = 'left'
            elif direction.lower() in ['r','ri','riht','right','rht','rt']:
                direction = 'right'
            else:
                raise ValueError
        except:
            logger.error('overlap direction must be either left or right')

        if not bool(self.im.overlap):
            if n_tiles > 1 and overlap > 0:
                tiles = []
                for t in range(n_tiles+1):
                    if direction == 'left':
                        cols = slice(2048*t+overlap,2048*(t+1))                 #initial columns are cropped from subsequent tiles
                        tiles.append(self.im.sel(col=cols))
                    elif direction == 'right':
                        cols = slice(2048*t,(2048-overlap)*(t+1))               #end columns are cropped from subsequent tiles
                        tiles.append(self.im.sel(col=cols))
                im = xr.concat(tiles, dim = 'col')
                im.attrs['overlap'] = overlap

                self.im = im
        else:
            logger.info('Overlap already removed')

    @staticmethod
    def save_jpeg(save_path, im, downscale=4):

        save_path = Path(save_path).with_suffix('.jpg')

        mid_log = log(0.5*255)
        
        jpegim = im.coarsen(row=downscale, col=downscale, boundary='trim').mean().astype('int16')

        # compute background px value = mode + std
        flat_ch = jpegim.data.flatten()
        hist_counts = da.bincount(flat_ch).compute()
        std = flat_ch.std().compute()
        mode = hist_counts.argmax()
        bg = std + mode

        # compute max as 99.75% px value
        max_px = da.percentile(flat_ch, 99.75).compute()

        # Compute mean and gamma correction 
        mean = flat_ch[flat_ch > bg].mean().compute()
        # if mean = 1, log(mean)=0, and ZeroDivisionError
        # limit mean to 10 to prevent extreme gamma values
        if mean < 10:
            mean = 10
        gamma =  mid_log / log(mean)

        # min / max normalize
        jpegim = (jpegim - bg) / (max_px - bg)
        jpegim = jpegim.clip(min = 0, max = 1)

        # gamma correct
        jpegim = (255 * (jpegim ** gamma)).astype('uint8')

        # write image
        imageio.imwrite(save_path, jpegim)


    def preview_jpeg(self, image_path=None, downscale=4, sel={}, im_name=''):
        
        if image_path is None:
            image_path = getcwd()
        image_path = Path(image_path)
        
        
        # Select channel/cycles to make previews
        im = self.im
        if len(sel) > 0:
            im = self.im.sel(sel)

        # zmax project
        if 'obj_step' in im.dims:
            im = im.max(dim='obj_step')

        if 'cycle' in im.dims and 'channel' in im.dims:
            for cy in im.cycle:
                for ch in im.channel:
                    im_name = image_path / f'{im.name}_r{cy}_ch{ch}.jpg'
                    self.save_jpeg(im_name, im.sel(cycle=cy, channel=ch))
        elif 'marker' in im.dims:
            for m in im.marker:
                im_name = image_path / f'{im.name}_{m}.jpg'
                self.save_jpeg(im_name, im.sel(marker=m))
        elif 'channel' in im.dims:
            for m in im.channel:
                im_name = image_path / f'{im.name}_{m}.jpg'
                self.save_jpeg(im_name, im.sel(marker=m))
        elif len(im.dims) == 2:
            im_name = image_path / f'{im.name}_{im_name}.jpg'
            self.save_jpeg(im_name, im)
                    



    def downscale(self, scale=None):
        if scale is None:
            size_Mb = self.im.size*16/8/(1e6)
            scale = int(2**round(log2(size_Mb)-10))

        if scale > 256:
            scale = 256

        if scale > 0:
            self.im = self.im.coarsen(row=scale, col=scale, boundary='trim').mean()
            old_scale = self.im.attrs['scale']
            self.im.attrs['scale']=scale*old_scale


    def crop_section(self, bound_box):
        """Return cropped full dataset with intact pixel groups.

           **Parameters:**
            - bound_box (list): Px row min, px row max, px col min, px col max

           **Returns:**
            - dataset: Row and column cropped full dataset
            - int: Initial pixel group index of dataset

        """

        if bound_box.shape[1] >= 2:
            bound_box = bound_box[:,-2:]

        nrows = len(self.im.row)
        ncols = len(self.im.col)
        #pixel group scale
        pgs = int(256/self.im.attrs['scale'])
        #tile scale
        ts = int(pgs*8)

        row_min = int(round(bound_box[0,0]))
        if row_min < 0:
            row_min = 0

        row_max = int(round(bound_box[1,0]))
        if row_max > nrows:
            row_max = nrows

        col_min = bound_box[0,1]
        if col_min < 0:
            col_min = 0
        col_min = int(floor(col_min/pgs)*pgs)

        col_max = bound_box[2,1]
        if col_max > ncols:
            col_max = ncols
        col_max = int(ceil(col_max/pgs)*pgs)

        group_index = floor((col_min + self.im.first_group*pgs)%ts/ts*8)

        self.im = self.im.sel(row=slice(row_min, row_max),
                              col=slice(col_min, col_max))
        self.im.attrs['first_group'] = group_index



    def save_zarr(self, save_path, show_progress = True, name=None, **kwargs):
        """Save all sections in a zipped zarr store.

           Note that coordinates for unused dimensions are not saved.

           **Parameters:**
            - save_path (path): directory to save store

        """

        save_path = Path(save_path)

        if save_path.is_dir():

            if name is None:
                save_name = save_path/f'{self.im.name}.zarr'
            else:
                save_name = save_path/f'{name}.zarr'
            # Remove coordinate for unused dimensions
            for c in self.im.coords.keys():
                if c not in self.im.dims:
                    self.im = self.im.reset_coords(names=c, drop=True)

            if show_progress:
                with ProgressBar() as pbar:
                    output = self.im.to_dataset().to_zarr(save_name, **kwargs)
            else:
                output = self.im.to_dataset().to_zarr(save_name, **kwargs)


            # save attributes
            with open(save_name.with_suffix('.yaml'), 'w') as f:
                yaml.dump(self.im.attrs, f)

            return output

        else:
            raise NotADirectoryError(f'{save_path} is not an existing directory')
        
    @staticmethod
    def read_xr_attrs(path):

        attrs = {}
        # Old method of reading xarray zarr store attributes 
        if path.with_suffix('.attrs').exists():

            with open(path.with_suffix('.attrs')) as f:
                for line in f:
                    items = line.split()
                    if len(items) == 2:
                        try:
                            value = int(items[1])
                        except ValueError: 
                            value = items[1]
                    else:
                        value = ''
                    attrs[items[0]] = value

        # New method readying from yaml 
        if path.with_suffix('.yaml').exists():
            with open(path.with_suffix('.yaml')) as f:
                attrs = yaml.safe_load(f)
        
        return attrs

    @classmethod
    def open_zarr(cls, path, common='', **kwargs):
        """Open xarray or OME saved zarrs.
           **Parameters:**
           - path(str): Directory to open zarrs from
           - common(str): Common name to filter which zarrs will be opened

           **Returns:**
           - array: Labeled dataset of list of labeled dataset

        """

        files = get_image_files(path, 'zarr', common)
        images = []
        for f in files:
            try:             
                ome_metadata = zarr.open_group(f, mode = 'r').attrs["omero"]
                im = cls.open_ome_zarr(ome_metadata, path, **kwargs)
            except KeyError:
                im = cls.open_xr_zarr(path, **kwargs)      

            if isinstance(im, list):
                images =+ im 
            else:
                images.append(im)

        n_images = len(images)    
        if n_images > 1:
            return images
        elif n_images == 1:
            return images[0]

    @classmethod
    def open_xr_zarr(cls, path, common='', **kwargs):
        """Create labeled dataset from zarrs.

          
        
           **Parameters:**
           - path(str): Directory to open zarrs from
           - common(str): Common name to filter which zarrs will be opened

           **Returns:**
           - array: Labeled dataset of list of labeled dataset

        """

        files = get_image_files(path, 'zarr', common)
        images = []
        for f in files:
            try: 
                im_name = f.stem
                im = xr.open_zarr(f).to_array()
                im = im.squeeze().drop_vars('variable').rename(im_name)

                # add attributes
                attrs = cls.read_xr_attrs(f)
                if 'machine' not in attrs.keys():
                    attrs['machine'] = None
                if len(attrs) > 0:
                    im = im.assign_attrs(**attrs)

                hsim = cls(im, machine=im.attrs['machine'], **kwargs)
                hsim.files = f
                images.append(hsim)
            except Exception:
                logger.error(f'Could not open {f}')
                logger.error(traceback.format_exc())

        n_images = len(images)    
        if n_images > 1:
            return images
        elif n_images == 1:
            return images[0]


    @classmethod
    def open_RoughScan(cls, path, **kwargs):

        # Open RoughScan tiffs
        files = get_image_files(path, 'zarr', 'RoughScan')


        comp_sets = dict()
        for f in files:
            # Break up filename into components
            comp_ = f.stem
            for i, comp in enumerate(comp_):
                comp_sets.setdefault(i,set())
                comp_sets[i].add(comp)

        shape = imageio.imread(files[0]).shape
        lazy_arrays = [dask.delayed(imageio.imread)(f) for f in files]
        lazy_arrays = [da.from_delayed(x, shape=shape, dtype='int16') for x in lazy_arrays]


        # Organize images
        #0 channel, 1 RoughScan, 2 x_step, 3 obj_step
        fn_comp_sets = list(comp_sets.values())
        for i in [0,2]:
            fn_comp_sets[i] = [int(x[1:]) for x in fn_comp_sets[i]]
        fn_comp_sets = list(map(sorted, fn_comp_sets))
        remap_comps = [fn_comp_sets[0], [1], fn_comp_sets[2]]
        a = np.empty(tuple(map(len, remap_comps)), dtype=object)
        for f, x in zip(files, lazy_arrays):
            comp_ = f.stem
            channel = fn_comp_sets[0].index(int(comp_[0][1:]))
            x_step = fn_comp_sets[2].index(int(comp_[2][1:]))
            a[channel, 0, x_step] = x


        # Label array
        dim_names = ['channel', 'row', 'col']
        channels = [int(ch) for ch in fn_comp_sets[0]]
        coord_values = {'channel':channels}
        im = xr.DataArray(da.block(a.tolist()),
                               dims = dim_names,
                               coords = coord_values,
                               name = 'RoughScan')

        # Assign Attributes
        im = im.assign_attrs(first_group = 0, scale=1,
                             overlap=0, fixed_bg = 0)
        # Remove top header
        im = im.sel(row=slice(64,None))
        
        hsim = cls(im)
        hsim.files = files

        return hsim

    @classmethod
    def open_objstack(cls, obj_stack, **kwargs):

        dim_names = ['frame', 'channel', 'row', 'col']
        channels = [687, 558, 610, 740]
        frames = range(obj_stack.shape[0])
        coord_values = {'channel':channels, 'frame':frames}
        im = xr.DataArray(obj_stack.tolist(),
                               dims = dim_names,
                               coords = coord_values,
                               name = 'Objective Stack')

        im = im.assign_attrs(first_group = 0, scale=1,
                             overlap=0, fixed_bg = 0)
    
        return cls(im)

    @classmethod
    def open_tiffs(cls, path='', files=[], common='', **kwargs):
        """Create labeled dataset from tiffs.

           **Parameters:**
           - path: Directory where tiffs are located
           - common (str): Common image name

           **Returns:**
           - array: Labeled dataset

        """

        # Open tiffs
        if len(files) == 0:
            files = get_image_files(path, 'tiff', common)
        section_sets = dict()
        section_meta = dict()
        for f in files:
            # Break up filename into components
            comp_ = f.stem.split('_')
            if len(comp_) >= 6:
                section = comp_[2]
                # Add new section
                if section_sets.setdefault(section, dict()) == {}:
                    im = imageio.imread(f)
                    section_meta[section] = {'shape':im.shape,'dtype':im.dtype,'files':[]}

                for i, comp in enumerate(comp_):
                    # Add components
                    section_sets[section].setdefault(i, set())
                    section_sets[section][i].add(comp)
                    section_meta[section]['files'].append(f)
        images = []
        for s in section_sets.keys():
            # Lazy open images
            files = section_meta[s]['files']
            lazy_arrays = [dask.delayed(imageio.imread)(f) for f in files]
            shape = section_meta[s]['shape']
            dtype = section_meta[s]['dtype']
            lazy_arrays = [da.from_delayed(x, shape=shape, dtype=dtype) for x in lazy_arrays]

            # Organize images
            fn_comp_sets = list(section_sets[s].values())
            if len(comp_) == 6:
                comp_order = {'ch':0, 'AorB':1, 's':2, 'r':3, 'x':4, 'o':5}
            elif len(comp_) == 7:
                comp_order = {'ch':0, 'AorB':1, 's':2, 'r':3, 'i':4, 'x':5, 'o':6}
            int_comps = ['ch', 'r', 'x', 'o']
            for i in [comp_order[c] for c in comp_order.keys() if c in int_comps]:
                fn_comp_sets[i] = [int(x[1:]) for x in fn_comp_sets[i]]
                fn_comp_sets[i] = sorted(fn_comp_sets[i])
            if 'i' in comp_order.keys():
                i = comp_order['i']
                fn_comp_sets[i] = [int(x) for x in fn_comp_sets[i]]
                fn_comp_sets[i] = sorted(fn_comp_sets[i])
                remap_comps = [fn_comp_sets[0], fn_comp_sets[3], fn_comp_sets[4], fn_comp_sets[6], [1],  fn_comp_sets[5]]
                # List of sorted x steps for calculating overlap
                #x_steps = sorted(list(fn_comp_sets[5]), reverse=True)
                x_steps = fn_comp_sets[5]
            else:
                remap_comps = [fn_comp_sets[0], fn_comp_sets[3], fn_comp_sets[5], [1],  fn_comp_sets[4]]
                # List of sorted x steps for calculating overlap
                #x_steps = sorted(list(fn_comp_sets[4]), reverse=True)
                x_steps = fn_comp_sets[4]

            a = np.empty(tuple(map(len, remap_comps)), dtype=object)
            for f, x in zip(files, lazy_arrays):
                comp_ = f.stem.split('_')
                channel = fn_comp_sets[0].index(int(comp_[0][1:]))
                cycle = fn_comp_sets[3].index(int(comp_[3][1:]))
                co = comp_order['o']
                obj_step = fn_comp_sets[co].index(int(comp_[co][1:]))
                co = comp_order['x']
                x_step = fn_comp_sets[co].index(int(comp_[co][1:]))
                if 'i' in comp_order.keys():
                    co = comp_order['i']
                    image_i = fn_comp_sets[co].index(int(comp_[co]))
                    a[channel, cycle, image_i, obj_step, 0, x_step] = x
                else:
                    a[channel, cycle, obj_step, 0, x_step] = x

            # Label array
            if 'i' in comp_order.keys():
                dim_names = ['channel', 'cycle', 'image', 'obj_step', 'row', 'col']
                coord_values = {'cycle':fn_comp_sets[3], 'channel':fn_comp_sets[0],'image':fn_comp_sets[4], 'obj_step':fn_comp_sets[6]}
            else:
                dim_names = ['channel', 'cycle', 'obj_step', 'row', 'col']
                coord_values = {'cycle':fn_comp_sets[3], 'channel':fn_comp_sets[0], 'obj_step':fn_comp_sets[5]}
            try:
                im = xr.DataArray(da.block(a.tolist()),
                                       dims = dim_names,
                                       coords = coord_values,
                                       name = s[1:])
                im = im.assign_attrs(first_group = 0, scale=1, overlap=0, fixed_bg = 0)
                hsim = cls(im, files=files, **kwargs)
                images.append(hsim)
            except Exception:
                logger.error('Could not make create DataArray for {s}')
                logger.error(traceback.format_exc())


        n_images = len(images)
        if n_images > 1:
            return images
        elif n_images == 1:
            return images[0]

    
    def save_ome_zarr(self, dir_path, compute=False):

        """Write labeled dataset to ome zarr.

           **Parameters:**
           - dir_path: Directory where OME zarrs will be saved to
           - compute (bool): Compute on demand on delay computation

           **Returns:**
           - array: list of delayed write tasks. list will be empty if compute=True

        """

        
        dir_path = Path(dir_path)
        instrument_ = Instrument(microscope = Microscope(**self.config['Microscope']), 
                                 objectives = [Objective(**self.config['Objective'])])
        im = self.im

        # Channel OME metadata
        channels_ = []
        if 'channel' in self.im.dims:
            for ch in im['channel']:
                channels_.append(Channel(name=str(int(ch)), **self.config.get('channels')[int(ch)]))
            size_c_ = len(im.channel)
            dim_order = 'TCZYX'
            im = im.transpose('cycle', 'channel', 'obj_step', 'row', 'col')
        elif 'marker' in self.im.dims:
            for ch in im['marker']:
                channels_.append(Channel(name=f'{ch.values}', fluor = f'Cycle {str(ch.cycle.values)}',
                                         **self.config.get('channels')[int(ch.channel.values)]))
            size_c_ = len(im.marker)
            dim_order = 'CZYX'
        else:
            raise KeyError('Missing Channel Dimension')
        # if not isinstance(self.im, list):
        #     self.im = [self.im]

        # for im in self.im:
        
        pxs = Pixels(channels = channels_, 
                     dimension_order = 'XYZCT', #Place holder, actually dim order TCZYX
                     size_x = len(im.col),
                     size_y = len(im.row),
                     size_z = len(im.obj_step),
                     size_c = size_c_,
                     size_t = len(im.cycle),
                     tiff_data_blocks = [TiffData(first_z = im.obj_step.values[0], first_t = im.cycle.values[0])],
                     **self.config['Pixels']
                     )
        description =f"""first_cycle = {im.cycle[0]},
                        last_cycle = {im.cycle[-1]},
                        first_objstep = {im.obj_step[0]},
                        last_objstep = {im.obj_step[-1]},
                        int_objstep = {im.obj_step[1]-im.obj_step[0]}
                        """
        ome = OME()
        ome.images = [Image(name = im.name, pixels = pxs, description = description)]
        ome.instruments = [instrument_]
        ome.creator = __name__
        ome_dict = to_dict(ome)

        # Fix UnSerializable Fields
        # Color Object Not JSON Serializable
        nch = len(ome_dict['images'][0]['pixels']['channels'])
        for i in range(nch):
            color_val = ome_dict['images'][0]['pixels']['channels'][i]['color'].as_rgb_tuple()
            ome_dict['images'][0]['pixels']['channels'][i]['color'] = color_val
            
        # Pixel Order Object Not JSON Serializable
        ome_dict['images'][0]['pixels']['dimension_order'] = dim_order
        ome_dict['images'][0]['pixels']['type'] = ome_dict['images'][0]['pixels']['type'].value
        

        # write the image data
        try: 
            store_path = dir_path/f'{im.name}.zarr'
            logger.debug(f'Saving to {store_path}')
            makedirs(store_path, exist_ok=True)
            store = parse_url(store_path, mode="w").store
            root = zarr.group(store=store)
            delayed_ = write_image(image=im.data, group=root, compute=compute,
                                   axes=dim_order.lower(), storage_options={'chunks':[i[0] for i in self.im.chunks]})
            root.attrs["omero"] = ome_dict
            return delayed_
        except Exception:
            logger.error(traceback.format_exc())
            logger.error("Error writing ome_zarr")
            
            return False
        

    @classmethod
    def open_ome_zarr(cls, ome_metadata, path, common='', **kwargs):


        """Create labeled dataset from ome zarrs.

            use open_zarr instead!

           **Parameters:**
           - ome_metadata: Dictionary of OME metadata
           - path: Directory where OME zarrs are located
           - common (str): Common image name

           **Returns:**
           - array: Labeled dataset

        """

        files = get_image_files(path, 'zarr', common)
        images = []

        for fn in files:

            reader = Reader(parse_url(fn, mode="r"))
            # nodes may include images, labels etc
            nodes = list(reader())
            # first node will be the image pixel data
            image_node = nodes[0]
            im = image_node.data[0] # full res image data

            # Read OME Metadata
            im_name = ome_metadata['images'][0]['name']
            meta_dict = {}
            for field in ome_metadata['images'][0]['description'].split(','):
                fn, val = field.split('=')
                meta_dict[fn.strip()] = int(val)
            
            # Map channel, cycle, and obj_step coordinates
            coord_dict = {'C': [c['name'] for c in ome_metadata['images'][0]['pixels']['channels']],
                          'T': range(meta_dict['first_cycle'], meta_dict['last_cycle']+1),
                          'Z': range(meta_dict['first_objstep'], meta_dict['last_objstep']+1, meta_dict['int_objstep'])
                         }

            dim_map = {'X': 'col', 'Y':'row', 'Z': 'obj_step', 'T': 'cycle', 'C': 'channel'}

            dims_ = []; coords_ = {}
            # loop through dimensions ie XYZCT
            for d in ome_metadata['images'][0]['pixels']['dimension_order']: 
                d_name = dim_map[d]
                dims_.append(d_name)
                if d in 'CTZ':
                    coords_[d_name] = coord_dict[d]
            xr_im = xr.DataArray(data = im, coords = coords_, dims = dims_, name = im_name)
            xr_im.attrs['omero'] = ome_metadata
            xr_im.attrs['machine'] = ome_metadata['instruments'][0]['microscope']['serial_number'].split(',')[0].strip()
            hsim =  cls(xr_im, machine=xr_im.attrs['machine'], **kwargs)
            hsim.files = fn
            images.append(hsim)

        n_images = len(images)    
        if n_images > 1:
            return images
        elif n_images == 1:
            return images[0]
