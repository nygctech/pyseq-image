from pyseq_pipeline import pre

experiment_config_path = snakemake.config['experiment_config_path']
experiment_config = pre.utils.get_config(experiment_config_path)
image_path = experiment_config['image_path']

section_name = snakemake.input[0]
image = ia.get_HiSeqImages(image_path = image_path, common_name = section_name)

image.correct_background()

image.save_zarr(snakemake.output[0])
