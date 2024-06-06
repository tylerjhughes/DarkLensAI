import jax
import jax.numpy as jnp
import astropy.io.fits as fits
from astropy.nddata import CCDData
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import pandas as pd

def asymmetry_compare(objid):
    a = get_asymmetry_cutout(objid, None, use_smap=False, debug=False, stampsize=512)
    return a 

def asymmetry(image, size=40, segm=None, sigma=0.0, debug=False):
    if segm is None:
        segm = jnp.ones_like(image)
    # Find the brightest pixel near the middle
    middle = 30 # squared
    img_size = image.shape[0]
    copy = jnp.zeros_like(image)
    copy = copy.at[img_size//2-middle:img_size//2+middle+1, img_size//2-middle:img_size//2+middle+1].set(1)
    copy *= image * segm
    
    y2, x2 = jnp.where(copy==copy.max())
    x1, y1 = x2[0], y2[0]
    image = image * segm
    ref = image[y1-size:y1+size+1, x1-size:x1+size+1]
    segm = segm[y1-size:y1+size+1, x1-size:x1+size+1]
    N_pix = (segm * jnp.rot90(segm, k=2)).sum()
    if debug:
        print(x1, y1, size, ref.shape, N_pix)
    rot = jnp.rot90(ref, k=2)
    if debug:
        plt.imshow(ref-rot)
        plt.show()
        
    noise = jax.random.normal(jax.random.PRNGKey(0), shape=ref.shape) * sigma
    noise_correct_empirical = jnp.abs(noise-jnp.rot90(noise, k=2)).sum()/(ref.shape[0]*ref.shape[1])
    noise_correct = noise_correct_empirical * N_pix
    
    if debug:
        print("K", noise_correct)
    a = (0.5 * jnp.abs(noise_correct - jnp.abs(ref - rot).sum())) / ref.sum()
    return a

def get_asymmetry_cutout(catalog, objid, band, size=40, noise_correct=0, unit=u.uJy,
                        get_segmap=False, use_smap=True, noise_measured=None, stampsize=100, debug=False):
    if band is None:
        band = catalog.loc[objid, 'rband'].lower()
    hdu = fetch_cutout(objid, catalog.loc[objid], band, size=512)
    image = CCDData(hdu.data, unit=unit)
    if noise_measured is None:
        noise_measured = get_noise(band, catalog.loc[objid])
    cat, segm, segm_deblend, vmax, vmin, source, quality_flag = get_seg_cutout(image, band, noise_measured)
    img_size = image.shape[0]
    
    if use_smap:
        if segm.data[img_size//2, img_size//2] != 0:
            mask = segm.data == segm.data[img_size//2, img_size//2]
        else: 
            mask = segm.data == source.label
    else:
        mask = jnp.ones_like(image.data)
    return asymmetry(image.data, size=size, segm=mask, sigma=noise_measured[band]['noise_sigma'], debug=debug), mask.sum()

if __name__ == '__main__':
    
    import os

    # Define the directory containing the FITS files
    directory = 'data/cosmos/'

    # Create an empty list to store the paths
    fits_files = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with .fits
        if filename.endswith('.fits'):
            # Construct the full file path and add it to the list
            fits_files.append(os.path.join(directory, filename))

    assymetry_values = []

    for image_path in fits_files:
        hdulist = fits.open(image_path)
        image = hdulist[0].data
        assymetry_values.append(asymmetry(image, size=40, segm=None, sigma=0.0, debug=False))

    plt.hist(assymetry_values, bins=30)
    plt.savefig('asymmetry_values.png')
