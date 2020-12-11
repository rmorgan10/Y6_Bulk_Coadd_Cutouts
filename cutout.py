from astropy.io import fits
import numpy as np
import pandas as pd


# also TODO: PSFs

class CutoutProducer:
    def __init__(self, tilename, cutout_size):
        self.metadata_path = "/data/des81.b/data/stronglens/Y6_CUTOUT_METADATA/"
        self.metadata_suffix = ".tab.gz"
        self.tilename = tilename
        self.cutout_size= cutout_size
    
    def read_metadata(self):
        """
        Read the metadata for the tile into a Pandas DataFrame
        """
        filename = self.metadata_path + self.tilename + self.metadata_suffix
        self.metadata = pd.read_csv(filename, delim_whitespace=True)
        return
    
    def read_tile_image(self):
        #TODO
        pass
    
    
    def get_locations(self):
        """
        Store the coordintaes of each galaxy
        """
        if not hasattr(self, "metadata"):
            self.read_metadata()
            
        self.locations = np.array([(ra, dec) for ra, dec in zip(self.metadata['RA'].values,
                                                                self.metadata['DEC'].values)])
        return
        
    def get_coadd_ids(self):
        """
        Get an array of all coadd object ids
        """
        if not hasattr(self, "metadata"):
            self.read_metadata()
            
        self.coadd_ids = np.array(self.metadata['COADD_OBJECT_ID'].values, dtype=int)
        return
        
    
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


