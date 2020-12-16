import os
import sys

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import pytest
import unittest

sys.path.append('..')
from cutout import CutoutProducer

class TestRaise(unittest.TestCase):
    def test_import_raise(self):
        with self.assertRaises(OSError):
            dummy_cutout_producer = CutoutProducer(tilename='i_dont_exist', cutout_size=45, path='test_data/')

class TestCutoutProducer(object):
    def setup(self):
        self.cutout_producer = CutoutProducer(tilename='testtile', cutout_size=45, path='test_data/')

    def test_read_metadata(self):
        assert hasattr(self.cutout_producer, "metadata")
        assert isinstance(self.cutout_producer.metadata, pd.core.frame.DataFrame)


if __name__ == "__main__":
    pytest.main()


