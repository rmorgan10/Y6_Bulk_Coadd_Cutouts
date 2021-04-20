#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors:
#   - Robert Morgan (robert.morgan@wisc.edu)
#   - Jackson O'Donnell (jacksonhodonnell@gmail.com)
#   - Jimena Gonzalez (sgonzalezloz@wisc.edu)
#   - Keith Bechtol (kbechtol@wisc.edu)
#   - Erik Zaborowski (ezaborowski@uchicago.edu)
#   - Simon Birrer (sibirrer@gmail.com)

"""
Single-epoch deep-field cutout production
"""

import argparse
import glob
import os
from pathlib import Path
import sys

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd


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

    def __init__(self, field, ccd, season, cutout_size, outdir, test=False, maglim=90,
                 image_path = '/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/images/',
                 metadata_path = '/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/metadata/',
                 catalog_path = '/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/catalog/'):

        """
        Initialize a CutoutProducer.

        """
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.catalog_path = catalog_path 
        self.metadata_suffix = ""
        self.test = test
        self.maglim = maglim
        self.field = field
        self.ccd = ccd
        self.season = season
        self.cutout_size = cutout_size
        self.outdir = outdir
        self.pad = 3 * self.cutout_size


    def read_metadata(self):
        """
        Read the metadata for the observations into a Pandas DataFrame
        """
        filename = self.metadata_path + self.field.upper() + "_metadata.csv" + self.metadata_suffix
        if not os.path.exists(filename):
            raise IOError(f"{filename} not found")
            
        # Trim to relevant metadata
        df = pd.read_csv(filename)
        df['CCD'] = df['FILENAME'].str.extract("_c(.*?)_").values.astype(int)
        df = df[(df['SEASON'].values == self.season.upper()) &
                (df['CCD'].values == self.ccd)
                ].copy().reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"SEASON {self.season} and CCD {self.ccd} produced empty metadata")

        # Select best psf if two images exist
        self.metadata = {}
        nite_groups = df.groupby('NITE')
        for (nite, md) in nite_groups:
            flt_groups = md.groupby('FLT')
            mds_ = {}
            for (flt, md_) in flt_groups:
                mds_[flt] = md_.iloc[np.argmin(md_['FWHM'].values)]

            self.metadata[nite] = mds_


        del df
        if len(self.metadata) == 0:
            raise ValueError(f"Trimming by nite and filter produced empty metadata")

        # sort by increasing MJD
        self.nites = sorted(list(self.metadata.keys()))

        # trim if testing
        if self.test:
            self.metadata = {self.nites[0]: self.metadata[self.nites[0]], 
                             self.nites[1]: self.metadata[self.nites[1]]}
            self.nites = self.nites[0:2]

    def read_catalog(self):
        """
        Read the catalog and trim to necessary rows
        """
        filename = self.catalog_path + "deep_catalog.csv"
        if not os.path.exists(filename):
            raise IOError(f"{filename} not found")

        # Trim to relevant catalog
        df = pd.read_csv(filename)
        self.catalog = df[(df['CCD'].values == self.ccd) &
                          (df['FIELD'].values == 'SN-' + self.field.upper()) &
                          (df['MAG_AUTO_I'].values < self.maglim)
                        ].copy().reset_index(drop=True)
        del df
        if len(self.catalog) == 0:
            raise ValueError(f"CCD {self.ccd} and FIELD {self.field} produced an empty catalog")

        # set fill flags
        self.fill_flags = np.zeros(len(self.catalog), dtype=bool)

        # trim if testing
        if self.test:
            self.catalog = self.catalog.sample(25, random_state=3)
            self.fill_flags = self.fill_flags[0:25]


    def read_image(self, filename):
        """
        Open a fits file and return the image array and the WCS

        :param filename: (str) name of file to read
        :return: image: (np.Array) the image data contained in the file
        :return: wcs: (astropy.WCS) the wcs for the file
        """
        f = fits.open(filename, mode='readonly')
        image = f["SCI"].data

        # pad image with min value
        image = np.pad(image, self.pad, mode='minimum')

        wcs = WCS(f["SCI"].header)
        f.close()
        return image, wcs

    def get_locations(self):
        """
        Store the coordintaes of each galaxy

        :return: ras: (np.Array) all RA values of objects in the tile
        :return: decs: (np.Array) all DEC values of objects in the tile
        """
        if not hasattr(self, "catalog"):
            self.read_catalog()

        return self.catalog['RA'].values, self.catalog['DEC'].values

    def get_coadd_ids(self):
        """
        Get an array of all coadd object ids
        """
        if not hasattr(self, "catalog"):
            self.read_catalog()

        self.coadd_ids = self.catalog['COADD_OBJECT_ID'].values.astype(int)
        return

    def get_object_xy(self, wcs):
        """
        Get the x, y coordinates within the coadd tile

        :return: object_x: (np.Array) all x values of objects in the tile
        :return: object_y: (np.Array) all y values of objects in the tile
        """
        ras, decs = self.get_locations()

        # Get pixel of each location, rounding
        pixel_x, pixel_y = wcs.world_to_pixel(SkyCoord(ras, decs, unit='deg'))
        object_x, object_y = pixel_x.round().astype(int), pixel_y.round().astype(int)
        
        # add padding
        return object_x + self.pad, object_y + self.pad

    def cutout_objects(self, image, wcs):
        """
        Grab square arrays from image using the wcs

        :param image: (np.Array) the image data contained in the file
        :param wcs: (astropy.WCS) the wcs for the file
        """
        # Get index locations of all objects
        object_x, object_y = self.get_object_xy(wcs)

        # Shout if any object is outside more than half a cutout_size outside the image
        if not np.all((1 - self.cutout_size //2 < object_x) & 
                      (object_x < image.shape[1] + self.cutout_size //2 - 1) &
                      (1 - self.cutout_size //2 < object_y) &
                      (object_y < image.shape[0] + self.cutout_size //2 - 1)):

            raise ValueError('Some objects centered out of CCD')


        # Flag objects on edges of CCD
        self.get_fill_flags(object_x, object_y)

        # Make cutouts
        cutouts = np.empty((len(self.coadd_ids), self.cutout_size, self.cutout_size), dtype=np.double)
        for i, (x, y) in enumerate(zip(object_x, object_y)):
            cutouts[i] = self.single_cutout(image, (x, y), self.cutout_size)

        return cutouts

    def single_cutout(self, image, center, width=None):
        """
        Creates a single cutout from an image.

        :param image: 2D Numpy array
        :param center: 2-tuple of ints, cutout center
        :param width: Int, the size in pixels of the cutout

        :return: 2D Numpy array, shape = (width, width)
        """
        x, y = center
        if width is None:
            width = self.cutout_size
        if width > min(image.shape):
            raise ValueError('Requested cutout is larger than image size')
        if (width % 2) == 0:
            return image[y - width//2: y + width//2,
                         x - width//2: x + width//2]
        else:
            return image[y - width//2: y + width//2 + 1,
                         x - width//2: x + width//2 + 1]


    def combine_bands(self, g_filename, r_filename, i_filename, z_filename):
        """
        Get cutouts (both image and psf) from all bands and stack into one array

        :return: image_array: (np.Array) shape = (number of cutouts, number of bands,
                                                  cutout_size, cutout_size)
        """
        if not hasattr(self, "coadd_ids"):
            self.get_coadd_ids()

        image_array = np.empty((4, len(self.coadd_ids), self.cutout_size, self.cutout_size), dtype=np.double)

        # Cutout objects and track bands missing on a given nite
        missing_idx = []
        for i, filename in enumerate([g_filename, r_filename, i_filename, z_filename]):
            if filename == 'flag':
                missing_idx.append(i)
                continue
                
            # Open image file
            image, wcs = self.read_image(filename)

            # Cutout images
            image_cutouts = self.cutout_objects(image, wcs)
            image_array[i] = image_cutouts

        # If any missing bands on a nite are detected, fill with the median value
        fill_val = np.median(image_array)
        for idx in missing_idx:
            image_array[idx] = fill_val

        image_array = np.swapaxes(image_array, 0, 1)

        return image_array

    @staticmethod
    def scale_array_to_ints(arr):
        """
        Scale an array of floats to ints 0-65535 (16-bit) using a per-image min-max scaling

        Original (but slightly rounded) values are recoverable by
            recovered_arr = int_arr / 65535 * shifted_max[:,:,np.newaxis,np.newaxis] + original_min[:,:,np.newaxis,np.newaxis]

        :param arr: an 4-dimensional array of floats
        :return: int_arr: the scaled and rounded version of arr, same shape
        :return: original_min: an array of the original minimum value of arr, 2-d
        :return: shifted_max: an array of the original maximum value of arr, 2-d
        """
        original_min = np.min(arr, axis=(-1, -2))
        shifted_max = np.max(arr - original_min[:,:,np.newaxis,np.newaxis], axis=(-1, -2))
        shifted_max = np.where(shifted_max == 0, 1.0, shifted_max)


        int_arr = np.rint(
            (arr - original_min[:,:,np.newaxis,np.newaxis]) / 
            shifted_max[:,:,np.newaxis,np.newaxis] * 65535).astype(np.uint16) 

        return int_arr, original_min, shifted_max

    def format_filename(self, base_filename):
        return f"{self.image_path}{self.season}/{base_filename}.fz"

    def get_fill_flags(self, object_x, object_y):
        
        object_x_ = object_x - self.pad
        object_y_ = object_y - self.pad

        fill_flags = ~((object_x_ - self.cutout_size //2 > 0) & 
                       (object_x_ + self.cutout_size //2 < 2048) & 
                       (object_y_ - self.cutout_size //2 > 0) & 
                       (object_y_ + self.cutout_size //2 < 4096))

        self.fill_flags = self.fill_flags | fill_flags


    def cutout_all_epochs(self):
        """
        loop through epochs and stack cutouts, track and organize metadata

        output_shape = (objects, band, height, width)
        """
        output = {}
        for nite in self.nites:
            # Get filenames from metadata
            filenames = {'g': 'flag', 'r': 'flag', 'i': 'flag', 'z': 'flag'}
            for flt in "griz":
                if flt in self.metadata[nite]:
                    filenames[flt] = self.format_filename(self.metadata[nite][flt]['FILENAME'])

            # Cutout objects
            image_array = self.combine_bands(filenames['g'], filenames['r'], filenames['i'], filenames['z'])

            # Scale array
            image_array, img_min, img_scale = self.scale_array_to_ints(image_array)

            # Store images
            output[nite] = {'IMG': image_array,
                            'IMG_MIN': img_min,
                            'IMG_SCALE': img_scale,
                            'METADATA': self.metadata[nite]}

        self.catalog['FILL_FLAG'] = self.fill_flags.astype(int)
        output['MAGLIM'] = self.maglim
        output['CATALOG'] = self.catalog
        output['NITES'] = self.nites
        return output
                     
    def get_outfile_name(self):
        return f"{self.outdir}{self.field}_{self.season}_{self.ccd}.npy"
        
    def save_output(self, out_dict):
        outfile_name = self.get_outfile_name()
        np.save(outfile_name, out_dict, allow_pickle=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--field',
                        type=str,
                        help='Like X1, X3, C3, etc.',
                        required=True)
    parser.add_argument('--ccd',
                        type=int,
                        help='CCD number as integer',
                        required=True)
    parser.add_argument('--season',
                        type=str,
                        help='Like SV, Y1, Y2, etc.',
                        required=True)
    parser.add_argument('--outdir',
                        type=str,
                        help='Directory for results',
                        default='')
    parser.add_argument('--size',
                        type=int,
                        help='Side length (in px) of cutouts',
                        default=45)
    parser.add_argument('--test',
                        action='store_true',
                        help='Run on only 5 objects and 2 nites')
    parser.add_argument('--maglim',
                        type=float,
                        help='Faintest i-magnitude to include',
                        default=90)
    
    args = parser.parse_args()

    # Make a CutoutProducer for the tile
    cutout_prod = CutoutProducer(
        args.field, args.ccd, args.season, args.size, args.outdir, test=args.test,
        maglim = args.maglim)

    # Check that we can save the output at the end
    outfile_name = cutout_prod.get_outfile_name()
    try:
        Path(outfile_name).touch(exist_ok=True)
    except Exception:
        raise OSError(f"{outfile_name} cannot be created")    

    # Read metadata and catalog
    cutout_prod.read_metadata()
    cutout_prod.read_catalog()

    # Check that all the files exist
    for nite, flt_dict in cutout_prod.metadata.items():
        for md in flt_dict.values():
            if not os.path.exists(cutout_prod.format_filename(md['FILENAME'])):
                raise OSError(f"{cutout_prod.format_filename(md['FILENAME'])} not found")

    # Do the thing
    cutout_prod.save_output(cutout_prod.cutout_all_epochs())




