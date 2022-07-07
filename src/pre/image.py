from pyseq_pipeline.pre.baseimage import BaseImage

class RawHiSeqImage():
    def __init__(image_dir, section_name):
        super().__init__(BaseImage)

class HiSeqImage():
    def __init__(image):
        super().__init__(BaseImage)

class ObjStack():
    def __init__(obj_stack):
        super().__init__(BaseImage)

class RoughScan():
    def __init__(image_dir,):
        super().__init__(BaseImage)

class Generic():
    def __init__(image_dir,):
        super().__init__(BaseImage)

class Custom():
    def __init__(image_dir,):
        super().__init__(BaseImage)
