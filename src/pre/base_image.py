import xarray as xr
import dask.array as da
import imageio
from pathlib import Path

from . import utils


class BaseImage():
    """Base class for images taken with HiSeq.

       **Attributes:**
        - im (xarray): Xarray wrapped dask DataArray
        - config (config): Configfile for machine
        - machine: Name of HiSeq used to take image
        - channel_color (dict): Dictionary of colors to display each channel as
        - logger: Logger object to log communication with HiSeq and user.
        - viewer (napari): Napari viewer
        - logger: Logger object to log communication with HiSeq and user.
        - resolution: microns per pixel
        - x_spum: step per micron for x stage
    """

    def __init__():

        machine = utils.get_machine()

        self.im = None
        self.config = None
        self.machine = machine
        self.channel_color = {558:'blue', 610:'green', 687:'magenta', 740:'red'}
        self.logger = None
        self.viewer = None
        self.resolution = 0.375                                                 # um/px
        self.x_spum = 0.4096                                                    #steps per um

    def correct_background(self):

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
                message(self.logger, pre_msg+'Image already background corrected.')
            elif machine is None:
                message(self.logger, pre_msg+'Unknown machine')


    def register_channels(self, image=None):
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
            message(self.logger, 'Unknown machine')

        return img

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
            n_tiles = int(len(self.im.col)/2048)
        except:
            message(self.logger, 'overlap must be an integer')

        try:
            if direction.lower() in ['l','le','lef','left','lft','lt']:
                direction = 'left'
            elif direction.lower() in ['r','ri','riht','right','rht','rt']:
                direction = 'right'
            else:
                raise ValueError
        except:
            message(self.logger, 'overlap direction must be either left or right')

        if not bool(self.im.overlap):
            if n_tiles > 1 and overlap > 0:
                tiles = []
                for t in range(n_tiles):
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
            message(self.logger, 'Overlap already removed')



    def quit(self):
        if self.stop:
            self.app.quit()
            self.viewer.close()
            self.viewer = None



    def hs_napari(self, dataset):

        with napari.gui_qt() as app:
            viewer = napari.Viewer()
            self.viewer = viewer
            self.app = app

            self.update_viewer(dataset)
            start = time.time()

            # timer for exiting napari
            timer = QTimer()
            timer.timeout.connect(self.quit)
            timer.start(1000*1)

            @viewer.bind_key('x')
            def crop(viewer):
                if 'Shapes' in viewer.layers:
                    bound_box = np.array(viewer.layers['Shapes'].data).squeeze()
                else:
                    bound_box = np.array(False)

                if bound_box.shape[0] == 4:

                    #crop full dataset
                    self.crop_section(bound_box)
                    #save current selection
                    selection = {}
                    for d in self.im.dims:
                        if d not in ['row', 'col']:
                            if d in dataset.dims:
                                selection[d] = dataset[d]
                            else:
                                selection[d] = dataset.coords[d].values
                    # update viewer
                    cropped = self.im.sel(selection)
                    self.update_viewer(cropped)


    def show(self, selection = {}, show_progress = True):
        """Display a section from the dataset.

           **Parameters:**
            - selection (dict): Dimension and dimension coordinates to display

        """

        dataset  = self.im.sel(selection)

        if show_progress:
            with ProgressBar() as pbar:
                self.hs_napari(dataset)
        else:
            self.hs_napari(dataset)

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

    def update_viewer(self, dataset):

        viewer = self.viewer
        # Delete old layers
        for i in range(len(viewer.layers)):
            viewer.layers.pop(0)

        # Display only 1 layer if there is only 1 channel
        channels = dataset.channel.values

        if not channels.shape:
            ch = int(channels)
            message(self.logger, 'Adding', ch, 'channel')
            layer = viewer.add_image(dataset.values,
                                     colormap=self.channel_color[ch],
                                     name = str(ch),
                                     blending = 'additive')
        else:
            for ch in channels:
                message(self.logger, 'Adding', ch, 'channel')
                layer = viewer.add_image(dataset.sel(channel = ch).values,
                                         colormap=self.channel_color[ch],
                                         name = str(ch),
                                         blending = 'additive')


    def save_zarr(self, save_path, show_progress = True, name=None):
        """Save all sections in a zipped zarr store.

           Note that coordinates for unused dimensions are not saved.

           **Parameters:**
            - save_path (path): directory to save store

        """

        if not path.isdir(save_path):
            mkdir(save_path)

        if name is None:
            save_name = path.join(save_path,self.im.name+'.zarr')
        else:
            save_name = path.join(save_path,str(name)+'.zarr')
        # Remove coordinate for unused dimensions
        for c in self.im.coords.keys():
            if c not in self.im.dims:
                self.im = self.im.reset_coords(names=c, drop=True)

        if show_progress:
            with ProgressBar() as pbar:
                self.im.to_dataset().to_zarr(save_name)
        else:
            self.im.to_dataset().to_zarr(save_name)


        # save attributes
        f = open(path.join(save_path, self.im.name+'.attrs'),"w")
        for key, val in self.im.attrs.items():
            f.write(str(key)+' '+str(val)+'\n')
        f.close()
