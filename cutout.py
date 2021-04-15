#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors:
#   - Robert Morgan (robert.morgan@wisc.edu)
#   - Jackson O'Donnell (jacksonhodonnell@gmail.com)
#   - Jimena Gonzalez (sgonzalezloz@wisc.edu)
#   - Keith Bechtol (kbechtol@wisc.edu)
#   - Erik Zaborowski (ezaborowski@uchicago.edu)
#   - Simon Birrer (sibirrer@gmail.com)

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

    def __init__(self, field, ccd, season, cutout_size, outdir, test=False,
                 image_path = '/afs/hep.wisc.edu/home/ramorgan2/DeepTransientSims/Data/',
                 metadata_path = '/afs/hep.wisc.edu/home/ramorgan2/DeepTransientSims/Data/',
                 catalog_path = '/afs/hep.wisc.edu/home/ramorgan2/DeepTransientSims/Data/'):

        """
        Initialize a CutoutProducer.

        """
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.catalog_path = catalog_path 
        self.metadata_suffix = ""
        self.test = test
        self.field = field
        self.ccd = ccd
        self.season = season
        self.cutout_size = cutout_size
        self.outdir = outdir


    def read_metadata(self):
        """
        Read the metadata for the observations into a Pandas DataFrame
        """
        filename = self.metadata_path + self.field.lower() + "_metadata_seasons.tab" + self.metadata_suffix
        if not os.path.exists(filename):
            raise IOError(f"{filename} not found")
            
        # Trim to relevant metadata
        df = pd.read_csv(filename, delim_whitespace=True)
        df['CCD'] = df['FILENAME'].str.extract("_c(.*?)_").values.astype(int)
        df = df[(df['SEASON'].values == self.season.upper()) &
                (df['CCD'].values == self.ccd)
                ].copy().reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"SEASON {self.season} and CCD {self.ccd} produced empty metadata")

        # Drop nites that don't have all of griz, select best psf if two images exist
        self.metadata = {}
        nite_groups = df.groupby('NITE')
        for (nite, md) in nite_groups:
            flt_groups = md.groupby('FILTER')
            mds_ = {}
            for (flt, md_) in flt_groups:
                mds_[flt] = md_.iloc[np.argmin(md_['FWHM'].values)]
            if len(mds_) == 4:
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
                          (df['FIELD'].values == 'SN-' + self.field.upper())
                          ].copy().reset_index(drop=True)
        del df
        if len(self.catalog) == 0:
            raise ValueError(f"CCD {self.ccd} and FIELD {self.field} produced an empty catalog")

        # trim if testing
        if self.test:
            self.catalog = self.catalog.sample(5)


    def read_image(self, filename):
        """
        Open a fits file and return the image array and the WCS

        :param filename: (str) name of file to read
        :return: image: (np.Array) the image data contained in the file
        :return: wcs: (astropy.WCS) the wcs for the file
        """
        f = fits.open(filename, mode='readonly')
        image = f["SCI"].data
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
        return object_x, object_y

    def cutout_objects(self, image, wcs):
        """
        Grab square arrays from image using the wcs

        :param image: (np.Array) the image data contained in the file
        :param wcs: (astropy.WCS) the wcs for the file
        """
        # Get index locations of all objects
        object_x, object_y = self.get_object_xy(wcs)

        # Shout if any object is outside of tile
        if not np.all((0 < object_x) & (object_x < image.shape[1])
                      & (0 < object_y) & (object_y < image.shape[0])):
            raise ValueError('Some objects centered out of tile')


        # FIXME: If an object is too close to a tile edge, single_cutout will
        # return a misshapen cutout, and this will throw an error
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
        
        for i, filename in enumerate([g_filename, r_filename, i_filename, z_filename]):
            # Open image file
            image, wcs = self.read_image(filename)

            # Cutout images
            image_cutouts = self.cutout_objects(image, wcs)
            image_array[i] = image_cutouts

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

        int_arr = np.rint(
            (arr - original_min[:,:,np.newaxis,np.newaxis]) / 
            shifted_max[:,:,np.newaxis,np.newaxis] * 65535).astype(np.uint16) 

        return int_arr, original_min, shifted_max

    def format_filename(self, base_filename):
        return f"{self.image_path}{self.season}/{base_filename}.fz"

    def cutout_all_epochs(self):
        """
        loop through epochs and stack cutouts, track and organize metadata

        output_shape = (objects, band, height, width)
        """
        output = {}
        for nite in self.nites:
            # Get filenames from metadata
            g_filename = self.format_filename(self.metadata[nite]['g']['FILENAME'])
            r_filename = self.format_filename(self.metadata[nite]['r']['FILENAME'])
            i_filename = self.format_filename(self.metadata[nite]['i']['FILENAME'])
            z_filename = self.format_filename(self.metadata[nite]['z']['FILENAME'])

            # Cutout objects
            image_array = self.combine_bands(g_filename, r_filename, i_filename, z_filename)

            # Scale array
            image_array, img_min, img_scale = self.scale_array_to_ints(image_array)

            # Store images
            output[nite] = {'IMG': image_array,
                            'IMG_MIN': img_min,
                            'IMG_SCALE': img_scale,
                            'METADATA': self.metadata[nite]}

        output['CATALOG'] = self.catalog
        output['NITES'] = self.nites
        return output
                     
    def get_outfile_name(self):
        return f"{self.outdir}{self.field}_{self.season}_{self.ccd}.npy"
        
    def save_output(self, out_dict):
        outfile_name = self.get_outfile_name()
        np.save(outfile_name, out_dict, allow_pickle=True)


if __name__ == "__main__":
    #assert len(sys.argv) == 2, "Tilename must be given as a a command-line argument"
    #tilename = sys.argv[1]
    CUTOUT_SIZE = 45
    field = 'X1'
    ccd = 33
    season = 'Y1'
    #OUTDIR = "/data/des81.b/data/stronglens/Y6_CUTOUT_IMAGES/"

    # Make a CutoutProducer for the tile
    cutout_prod = CutoutProducer(field, ccd, season, CUTOUT_SIZE, outdir="", test=True)

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




