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
import sys

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import fitsio
import numpy as np
import pandas as pd

import galsim
from galsim.des.des_psfex import DES_PSFEx


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

    def __init__(self, tilename, cutout_size, psf_cutout_size, bands="griz",
                 metadata_path='/data/des81.b/data/stronglens/Y6_CUTOUT_METADATA/',
                 coadds_path='/data/des40.b/data/des/y6a2/coadd/image/',
                 psf_path='/data/des40.b/data/des/y6a2/coadd/image/'):

        """
        Initialize a CutoutProducer.

        :param tilename: (str) name of DES tile; something like 'DES0536-5457'
        :param cutout_size: (int) side length in pixels of desired cutouts
        :param psf_cutout_size: (int) side length in pixels of desired PSF cutouts
        :param bands: (str) bands to include; something like "griz" or "grizY"
        :param coadds_path: (str) path relative to the COADD images (organized as tiles)
        :param psf_path: (str) path relative to the PSF files (organized as tiles)
        """
        self.metadata_path = metadata_path
        self.coadds_path = coadds_path 
        self.psf_path = psf_path
        self.metadata_suffix = ".tab.gz"
        self.tilename = tilename
        self.cutout_size = cutout_size
        self.psf_cutout_size = psf_cutout_size
        self.bands = bands
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

        guess = os.path.join(self.coadds_path, self.tilename,
                             f'{self.tilename}_r*_{band}.fits.fz')

        matches = glob.glob(guess)

        if len(matches) > 1:
            return ValueError('error - more than one possible coadd')
        elif not matches:
            raise ValueError('no images found')

        return matches[0]

    def get_tile_psf_filename(self, band):
        """
        construct the filepath to the coadd tile PSF model

        :param band: (str) one of ('g', 'r', 'i', 'z', 'Y')
        :return: path: (str) absolute path to tile
        """
        guess = os.path.join(self.coadds_path, self.tilename,
                             f'{self.tilename}_r*_{band}_psfcat.psf')

        matches = glob.glob(guess)

        if len(matches) > 1:
            return ValueError('error - more than one possible coadd')
        elif not matches:
            raise ValueError('no images found')

        return matches[0]

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

    def get_tile_wcs(self):
        """
        Read the wcs information from the file header. This method is not used to
        obtain the wcs that other functions in this class use. This information 
        is written directly to the output fits file so that the original WCS info
        is a part of the final cutous.

        :return: wcs_table: table of wcs info to be written to output file
        """
        DTYPE = [ (f"CTYPE{i}",'S8') for i in [1,2]] +\
                [ (f"DCRPIX{i}",float) for i in [1,2]] + \
                [ (f"CRPIX{i}", float) for i in [1,2]]  + \
                [ (f"CRVAL{i}", float) for i in [1,2]]  + \
                [ (f"CD{j}_{i}",float) for i in [1,2] for j in [1,2]] #  + \
                #[ (f"PV{j}_{i}",float) for i in range(11) for j in [1,2]]
        DTYPES = [ (n, (d, len(self.bands))) for n, d in DTYPE]

        wcs_table = np.zeros(1, dtype=DTYPES)
        wcs_table.fill(np.nan)
        for idx, band in enumerate(self.bands):
            tile_filename = self.get_tile_filename(band)
            hdr = fitsio.read_header(tile_filename, ext='SCI')
            for name, dt in DTYPE:
                try:
                    wcs_table[name][:,idx] = hdr[name]
                except KeyError:
                    pass

        return wcs_table

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

    def read_psf(self, band):
        """
        Read a psf into memory

        :param band: (str) band to target
        :return: a DES_PSFEx instance
        """
        filename = self.get_tile_psf_filename(band)
        psf = DES_PSFEx(filename)
        setattr(self, f"psf_samp_{band}", psf.getSampleScale())
        return psf
    
    def cutout_psfs(self, psf, wcs):
        """
        Grab square PSF cutout images

        :param psf: a DES_PSFEx instance 
        :param wcs: (astropy.WCS) the wcs for the tile

        :return: 3D Numpy array
        """
        object_x, object_y = self.get_object_xy(wcs)
        
        psf_cutouts = np.empty((len(self.coadd_ids), self.psf_cutout_size, self.psf_cutout_size), dtype=np.double)

        for i, (x, y) in enumerate(zip(object_x, object_y)):
            pos = galsim.PositionI(x,y) 
            psfimg = psf.getPSFArray(pos)
            center = (psfimg.shape[0] // 2, psfimg.shape[1] // 2)
            psfimg = self.single_cutout(psfimg, center, self.psf_cutout_size)
            psf_cutouts[i] = psfimg

        return psf_cutouts

    def combine_bands(self):
        """
        Get cutouts (both image and psf) from all bands and stack into one array

        :return: image_array: (np.Array) shape = (number of cutouts, number of bands,
                                                  cutout_size, cutout_size)
        """
        if not hasattr(self, "coadd_ids"):
            self.get_coadd_ids()

        image_array = np.empty((len(self.bands), len(self.coadd_ids), self.cutout_size, self.cutout_size), dtype=np.double)
        psf_array = np.empty((len(self.bands), len(self.coadd_ids), self.psf_cutout_size, self.psf_cutout_size), dtype=np.double)
        
        for i, band in enumerate(self.bands):
            # Open image file
            image, wcs = self.read_tile_image(band)
            psf = self.read_psf(band)

            # Cutout images
            image_cutouts = self.cutout_objects(image, wcs)
            image_array[i] = image_cutouts

            # Cutout psfs
            psf_cutouts = self.cutout_psfs(psf, wcs)
            psf_array[i] = psf_cutouts
            
        image_array = np.swapaxes(image_array, 0, 1)
        psf_array = np.swapaxes(psf_array, 0, 1)

        return image_array, psf_array

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


    def produce_cutout_file(self, image_array, psf_array, out_dir=''):
        """
        Organize cutout data into an output file

        :param image_array: (np.array) contains all images. Has
                            shape = (len(coadd_ids), num_bands, cutout_size, cutout_size)
        :param psf_array: (np.array) contains all psfs. Has
                            shape = (len(coadd_ids), num_bands, psf_size, psf_size) 
        :param out_dir: (str) path to out directory
           - leave as default '' for current directory

        """
        # Scale the image and psf arrays
        image_array, img_min, img_scale = self.scale_array_to_ints(image_array)
        psf_array, psf_min, psf_scale = self.scale_array_to_ints(psf_array)

        # Make an empty PRIMARY HDU
        primary = fits.PrimaryHDU()

        # Make the IMAGE HDU
        image = fits.ImageHDU(image_array, name="IMAGE")

        # Make the PSF HDU
        header_dict = {}
        for b in self.bands:
            header_dict[f'PSFSAMP{b}'] = eval(f"self.psf_samp_{b}")
        psf = fits.ImageHDU(psf_array, name="PSF", header=fits.Header(header_dict))

        # Make the INFO HDU
        if not hasattr(self, "coadd_ids"):
            self.get_coadd_ids()
        data_info = {'ID':        {'NAME': 'self.coadd_ids', 'DTYPE': 'S14'},
                     'IMG_MIN':   {'NAME': 'img_min',        'DTYPE': (float, len(self.bands))},
                     'IMG_SCALE': {'NAME': 'img_scale',      'DTYPE': (float, len(self.bands))},
                     'PSF_MIN':   {'NAME': 'psf_min',        'DTYPE': (float, len(self.bands))},
                     'PSF_SCALE': {'NAME': 'psf_scale',      'DTYPE': (float, len(self.bands))}}
        dtypes = [(name, info['DTYPE']) for name, info in data_info.items()]
        table = np.zeros(len(self.coadd_ids), dtype=dtypes)
        for name, info in data_info.items():
            table[name] = eval(info['NAME'])
        info = fits.BinTableHDU(data=table, name='INFO')

        # Make the WCS HDU
        wcs = fits.BinTableHDU(data=self.get_tile_wcs(), name='WCS')

        # Write the file
        hdu_list = fits.HDUList([primary, image, psf, info, wcs])
        if not out_dir.endswith('/') and out_dir != '':
            out_dir += '/'
        hdu_list.writeto(f'{out_dir}{self.tilename}.fits', overwrite=True)
        return


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Tilename must be given as a a command-line argument"
    tilename = sys.argv[1]
    CUTOUT_SIZE = 45
    PSF_CUTOUT_SIZE = 25
    BANDS = "griz"
    #OUTDIR = "/data/des81.b/data/stronglens/Y6_CUTOUT_IMAGES/"
    OUTDIR = "/data/des81.b/data/stronglens/PRODUCTION/Y6_CUTOUT_IMAGES/full_tile_test/"

    # Make a CutoutProducer for the tile
    cutout_prod = CutoutProducer(tilename, CUTOUT_SIZE, PSF_CUTOUT_SIZE, bands=BANDS)

    # Quit if missing files
    for band in cutout_prod.bands:
        tile_path = cutout_prod.get_tile_filename(band)
        assert os.path.exists(tile_path), "Coadd image should exist"
        psf_path = cutout_prod.get_tile_psf_filename(band)
        assert os.path.exists(psf_path), "PSF file should exist"
        
    # Produce the cutouts
    cutout_prod.read_metadata()
    image_array, psf_array = cutout_prod.combine_bands()
    cutout_prod.produce_cutout_file(image_array, psf_array, out_dir=OUTDIR)


