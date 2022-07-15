from pyseq import image_analysis as ia
from pre import utils 
from os.path import join 

experiment_config = utils.get_config(snakemake.input[0])
exp_dir = snakemake.config['experiment_directory']
image_path = snakemake.config.get('image_path',experiment_config['experiment']['image path'])
image_path = join(exp_dir, image_path)

section_name = snakemake.params.section
print(image_path)
print(section_name)
image = ia.get_HiSeqImages(image_path = image_path, common_name = section_name)

image.correct_background()

image.save_zarr(snakemake.params.save_path)
