import configparser
from pathlib import Path
import yaml
import logging

def get_logger(name = None, filehandler = None, consolehandler = None):

    if name is  None:
        name = __name__

    if filehandler is None and consolehandler is None:
        raise ValueError('Specify filehandler or consolehandler')

    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if filehandler is not None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filehandler)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if consolehandler is not None:
        # create console handler with a higher log level
        ch = logging.StreamHandler(consolehandler)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger




def get_config(config_path):

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.exists():
        raise OSError('config does not exist')

    if config_path.suffix == '.cfg':
        # Create and read experiment config
        config = configparser.ConfigParser()
        config.read(config_path)

        return config

    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config



def get_machine():

    machine_info_paths = [Path.home()/'.config'/'pyseq2500'/'machine_info.yaml',
                          Path.home()/'.config'/'pyseq2500'/'machine_info.cfg',
                          Path.home()/'.pyseq2500'/'machine_info.yaml',
                          Path.home()/'.pyseq2500'/'machine_info.cfg'
                          ]
    machine = None
    for p in machine_info_paths:
        machine_info = get_config(p)
        if machine_info is not None:
            machine = machine_info.defaults['name']
            break

    return machine


def parse_sections(config):
    '''Return names of sections from experiment config file.'''

    if isinstance(config, str):
        config = get_config(config)

    return config.options('sections')

def get_exp_name(config):

    if isinstance(config, str):
        config = get_config(config)

    return config.get('experiment', 'experiment name')
