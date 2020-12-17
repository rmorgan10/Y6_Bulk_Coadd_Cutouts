#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors:
#   - Robert Morgan (robert.morgan@wisc.edu)
#   - Jackson O'Donnell (jacksonhodonnell@gmail.com)
#   - Jimena Gonzalez (sgonzalezloz@wisc.edu)

import os
import sys

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd

# TODO: tests
# also TODO: PSFs

class CutoutProducer:
    """
    Class to produce cutouts of all objects on a specified DES tile. 
    
    Based on the provided tilename, the CutoutProducer locates and reads a 
    compressed text file (referred to as metadata) specific to the tile. The 
    metadata contains the object COADD_OBJECT_ID, RA, and DEC (among other 
    properties) in a tabular format.
    
    The objects in the tile are then located using the FITS header of the 
    DES tile file and the RA, DEC of each object.
    
    Image cutouts of the same size in each band are then stored in an array
    of size (num_objects, num_bands, cutout_size, cutout_size) and saved to a
    FITS file.
    
    """
    def __init__(self, tilename, cutout_size):
        """
        Initialize a CutoutProducer.
        
        :param tilename: (str) name of DES tile; something like 'DES0536-5457'
        :param cutout_size: (int) side length in pixels of desired cutouts
        """
        self.metadata_path = "/data/des81.b/data/stronglens/Y6_CUTOUT_METADATA/"
        self.metadata_suffix = ".tab.gz"
        self.tilename = tilename
        self.cutout_size = cutout_size
        self.read_metadata()
        return

    def read_metadata(self):
        """
        Read the metadata for the tile into a Pandas DataFrame
        """
        filename = self.metadata_path + self.tilename + self.metadata_suffix
        if not os.path.exists(filename):
            raise IOError(f"{self.tilename} is not a valid tilename")

        self.metadata = pd.read_csv(filename, delim_whitespace=True)
        return

    def get_tile_filename(self, band):
        """
        construct the filepath to the coadd tile

        :param band: (str) one of ('g', 'r', 'i', 'z', 'Y')
        :return: path: (str) absolute path to tile
        """
        raise NotImplementedError("Someone needs to do this")


    def read_tile_image(self, band):
        """
        Open a fits file and return the image array and the WCS

        :param band: (str) one of ('g', 'r', 'i', 'z', 'Y')
        :return: image: (np.Array) the image data contained in the file
        :return: wcs: (astropy.WCS) the wcs for the file
        """
        tile_filename = self.get_tile_filename(band)

        f = fits.open(tile_filename, mode='readonly')
        image = f[1].data
        wcs = WCS(f[1].header)
        f.close()
        return image, wcs


    def get_locations(self):
        """
        Store the coordintaes of each galaxy

        :return: ras: (np.Array) all RA values of objects in the tile
        :return: decs: (np.Array) all DEC values of objects in the tile
        """
        if not hasattr(self, "metadata"):
            self.read_metadata()

        return self.metadata['RA'].values, self.metadata['DEC'].values

    def get_coadd_ids(self):
        """
        Get an array of all coadd object ids
        """
        if not hasattr(self, "metadata"):
            self.read_metadata()

        self.coadd_ids = np.array(self.metadata['COADD_OBJECT_ID'].values, dtype=int)
        return

    def cutout_objects(self, image, wcs):
        """
        Grab square arrays from image using the wcs

        :param image: (np.Array) the image data contained in the file
        :param wcs: (astropy.WCS) the wcs for the file
        """
        ras, decs = self.get_locations()

        # Get pixel of each location, rounding
        pixel_x, pixel_y = wcs.world_to_pixel(SkyCoord(ras, decs, unit='deg'))
        object_x, object_y = pixel_x.round().astype(int), pixel_y.round().astype(int)

        # Shout if any object is outside of tile
        if not np.all((0 < object_x) & (object_x < image.shape[0])
                    & (0 < object_y) & (object_y < image.shape[1])):
            raise ValueError('Some objects centered out of tile')

        # FIXME: If an object is too close to a tile edge, single_cutout will
        # return a misshapen cutout, and this will through an error
        cutouts = np.empty((len(ras), self.cutout_size, self.cutout_size), dtype=np.double)
        for i, (x, y) in enumerate(zip(object_x, object_y)):
            cutouts[i] = single_cutout(image, (x, y), width)
        return cutouts

    def single_cutout(self, image, center):
        """
        Creates a single cutout from an image.

        :param image: 2D Numpy array
        :param center: 2-tuple of ints, cutout center
        :param width: Int, the size in pixels of the cutout

        :return: 2D Numpy array, shape = (width, width)
        """
        x, y = center
        width = self.cutout_size
        if (width % 2) == 0:
            return image[x - width//2 : x + width//2,
                         y - width//2 : y + width//2]
        return image[x - width//2 : x + width//2 + 1,
                     y - width//2 : y + width//2 + 1]

    def combine_bands(self, bands=('g', 'r', 'i', 'z')):
        """
        Get cutouts from all bands and stack into one array

        :return: image_array: (np.Array) shape = (number of cutouts, number of bands,
                                                  cutout_size, cutout_size)
        """
        if not hasattr(self, "coadd_ids"):
            self.get_coadd_ids()

        image_array = np.empty((len(bands), len(self.coadd_ids), self.cutout_size, self.cutout_size), dtype=np.double)

        for i, band in enumerate(bands):
            image, wcs = self.read_tile_image(band)
            cutouts = self.cutout_objects(image, wcs)
            image_array[i] = cutouts

        image_array = np.swapaxes(image_array, 0, 1)

        return image_array

    def produce_cutout_file(self, image_array, out_dir=''):
        """
        Organize cutout data into an output file

        :param image_array: (np.array) contains all images. Has
                            shape = (len(coadd_ids), 4, 65, 65)
                             - 4 bands = g,r,i,z
                             - 65 x 65 = image height x width
                                - doesn't have to be 65 x 65
        :param out_dir: (str) path to out directory
           - leave as default '' for current directory

        """
        if not hasattr(self, "coadd_ids"):
            self.get_coadd_ids()

        # Make the ID HDU
        col = fits.Column(name='COADD_ID', format='int16', array=self.coadd_ids)
        cols = fits.ColDefs([col])
        header = fits.BinTableHDU.from_columns(cols, name='IDS')

        # Make the IMAGE HDU
        image = fits.PrimaryHDU(image_array)

        # Write the file
        hdu_list = fits.HDUList([image, header])
        if not out_dir.endswith('/') and out_dir != '':
            out_dir += '/'
        hdu_list.writeto(f'{out_dir}{self.tilename}.fits', overwrite=True)
        return


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Tilename must be given as a a command-line argument"
    tilename = sys.argv[1]
    CUTOUT_SIZE = 45

    cutout_prod = CutoutProducer(tilename, CUTOUT_SIZE)

    raise NotImplementedError("Someone needs to do this")
