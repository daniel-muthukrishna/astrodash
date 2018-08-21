import numpy as np
from astropy.io import fits


def make_fits_file(inputFilename, outFilename):
    """ Code adapted from Mat Smith's 'spec_convert_global.py"""

    spec = np.loadtxt(inputFilename, dtype=[('wave', np.double), ('flux', np.double), ('err', np.double)])
    a = [np.where((spec['wave'] > 0) & (spec['wave'] < 200000))]
    prihdr = fits.Header()
    pix1 = a[0][0][0]
    pix2 = a[0][0][1]
    prihdr['CRVAL1'] = spec['wave'][pix1]
    prihdr['CDELT1'] = spec['wave'][pix2]-spec['wave'][pix1]
    prihdr['CRPIX1'] = 1.
    prihdr['CTYPE1'] = 'Wavelength'
    prihdr['CUNIT1'] = 'Angstroms'

    # The first image in the fits file is the flux
    hdu = fits.PrimaryHDU(spec['flux'][a], header=prihdr)
    hdulist = fits.HDUList(hdu)
    # The first extension is the variance: note the errors are squared: test this.
    hdulist.append(fits.ImageHDU((spec['err'][a])**2,name='variance'))
    # The next extension tells MarZ that we are looking at a transient.
    a1 = np.array(['DES16X3eww'])
    a2 = np.array(['P'])
    a3 = np.array(['Transient'])
    col1 = fits.Column(name='NAME', format='80A', array=a1)
    col2 = fits.Column(name='TYPE', format='1A', array=a2)
    col3 = fits.Column(name='COMMENT', format='80A', array=a3)
    cols = fits.ColDefs([col1,col2,col3])
    tbhdu = fits.BinTableHDU.from_columns(cols,name='FIBRES')
    hdulist.append(tbhdu)
    #  Finally we have the wavelength. This is in the header too, but it's good to heave it here for completeness.
    hdulist.append(fits.ImageHDU(spec['wave'][a],name='wavelength'))
    # Output it as a fits file
    hdulist.writeto(outFilename, overwrite=True)


if __name__ == "__main__":
    make_fits_file('/home/daniel/Documents/DES_Y4_Spectra/VLT/DES16C2aiy_all.txt', '/home/daniel/Desktop/outFits.fits')


