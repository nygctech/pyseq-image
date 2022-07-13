import configparser
from pathlib import Path



def get_config(config_path):

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if config_path.exists():
        # Create and read experiment config
        config = configparser.ConfigParser()
        config.read(config_path)

        return config
    else:
        return None


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


def parse_sections(file_path):
    '''Return names of sections from experiment config file.'''

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.suffix == '.cfg':
        config = configparser.ConfigParser()
        config.read(file_path)
        sections = config.options('sections')

    return sections
