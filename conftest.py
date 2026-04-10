import logging

import pytest

pytest_plugins = ['demo.fixtures_demo']

logging.getLogger("fsspec.local").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.writer").setLevel(logging.INFO)
logging.getLogger("ome_zarr.io").setLevel(logging.INFO)
logging.getLogger("ome_zarr.format").setLevel(logging.INFO)
logging.getLogger("numcodecs").setLevel(logging.INFO)
