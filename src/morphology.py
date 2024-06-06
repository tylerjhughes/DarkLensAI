import numpy as np
import pandas as pd
import h5py
from photutils.centroids import centroid_com
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from lenstronomy.Util import data_util
import scipy

class Galaxy(object):
    def __init__(self, img):
        self.img = img
        self.radv, self.rad_array = self._radv()
        self.segmentation_map = self._segment(img)

    
    def _radv(self):
        xv, yv = np.meshgrid(np.arange(self.img.shape[2]), np.arange(self.img.shape[1]))
        xv = np.repeat(xv[None, :, :], self.img.shape[0], axis=0)
        yv = np.repeat(yv[None, :, :], self.img.shape[0], axis=0)
        x, y = np.array([centroid_com(image) for image in self.img]).T
        xv = xv.astype(float)
        rad_array = np.sqrt((xv - x[:, None, None])**2 + (yv - y[:, None, None])**2)
        del xv, yv, x, y
        return np.linspace(0.0000001, np.shape(self.img)[-1], 1000, dtype=np.float32), rad_array

    
    def _segment(self, img):
        delta_r = 1
        rad_array_exp = np.expand_dims(self.rad_array, axis=-1)
        radv_exp = np.expand_dims(self.radv, axis=0)
        pixels_at_r = ((rad_array_exp > radv_exp) & (rad_array_exp < radv_exp + delta_r))
        pixels_in_r = (rad_array_exp < radv_exp)
        del rad_array_exp, radv_exp
        flux_at_r = np.sum(img[..., None] * pixels_at_r, axis=(1, 2))
        flux_in_r = np.sum(img[..., None] * pixels_in_r, axis=(1, 2))
        mean_flux_at_r = flux_at_r / np.sum(pixels_at_r, axis=(1, 2))
        del pixels_at_r
        mean_flux_in_r = flux_in_r / np.sum(pixels_in_r, axis=(1, 2))
        del pixels_in_r, flux_at_r, flux_in_r
        ita = np.round(mean_flux_at_r / mean_flux_in_r, 5)
        ita[np.isnan(ita)] = 10000000
        mean_flux_at_r = mean_flux_at_r[np.arange(mean_flux_at_r.shape[0]), np.argmin(np.abs(ita - 0.2), axis=-1)]
        seg_map = np.where(img > mean_flux_at_r[:, None, None], 1, np.nan)
        del mean_flux_at_r, mean_flux_in_r, ita
        return seg_map
    
    
    def gini(self, segmap=False):

        if segmap == True:
            img = self.img*self.segmentation_map
        else:
            img = self.img

        img_flattened = np.reshape(img, (img.shape[0], -1), order='C')
        n = np.count_nonzero(~np.isnan(img_flattened), axis=1)
        mean_pixel = np.nanmean(img_flattened, axis=1)
        sorted_pixels = np.sort(img_flattened, axis=1)
        A = 1/(mean_pixel*n*(n-1))
        gini = [A[i] * np.sum((2*np.arange(1,n[i]+1) - n[i] - 1)*sorted_pixels[i][0:n[i]]) for i in range(len(n))]

        del img, img_flattened, n, mean_pixel, sorted_pixels, A

        return gini

    
    def concentration(self):
        rad_array_exp = np.expand_dims(self.rad_array, axis=-1)
        radv_exp = np.expand_dims(self.radv, axis=0)
        pixels_in_r = (rad_array_exp < radv_exp)
        flux_in_r = np.sum(self.img[...,None]*pixels_in_r, axis=(1, 2))
        max_flux = np.max(flux_in_r, axis=1)
        repeat_max_flux = np.repeat(max_flux[:, None], self.radv.shape[-1], axis=1)
        r_20 = np.where(flux_in_r < 0.2*repeat_max_flux, radv_exp, 0)
        r_20 = np.max(r_20, axis=1)
        r_80 = np.where(flux_in_r < 0.8*repeat_max_flux, radv_exp, 0)
        r_80 = np.max(r_80, axis=1)

        del rad_array_exp, radv_exp, pixels_in_r, flux_in_r, max_flux, repeat_max_flux

        return 5*np.log10(r_80/r_20)

    
    def asymmetry(self):
        img_rot = np.rot90(self.img, 2, axes=(1, 2))
        asymmetry = np.sum(np.abs(self.img - img_rot), axis=(1, 2))/np.sum(self.img, axis=(1, 2))

        del img_rot

        return asymmetry

    
    def m_20(self):
        img = self.img*self.segmentation_map
        m_tot = np.nansum(img*self.rad_array**2, axis=(1, 2))
        f_tot = np.nansum(img, axis=(1, 2))
        img_reshaped = np.reshape(img, (img.shape[0], img.shape[1]*img.shape[2]))
        rad_array_reshaped = np.reshape(self.rad_array, (self.rad_array.shape[0], self.rad_array.shape[1]*self.rad_array.shape[2]))
        sort_indices = np.argsort(img_reshaped, axis=-1)
        rad_array_sorted = np.take_along_axis(rad_array_reshaped, sort_indices, axis=-1)
        rad_array_sorted = rad_array_sorted[:, ::-1]
        img_sorted = np.take_along_axis(img_reshaped, sort_indices, axis=-1)[:, ::-1]
        cumsum_img = np.nancumsum(img_sorted, axis=-1)

        del img, img_reshaped, rad_array_reshaped, sort_indices

        return np.log10(np.nansum(np.where(cumsum_img < 0.2*f_tot[:, None], img_sorted*rad_array_sorted**2, 0), axis=-1)/m_tot)


def process_images():
    batch_size = 100
    results = []

    with h5py.File('data/dataset_source.h5', 'r') as f:
        total_images = f['CONFIGURATION_2_images'].shape[0]
        for i in range(0, total_images, batch_size):
            print(f'Processing batches: {i//batch_size + 1}/{total_images//batch_size}')
            images = np.array(f['CONFIGURATION_2_images'][i:i+batch_size, 0, :, :])
            gal_batch = Galaxy(images)
            gini = gal_batch.gini(segmap=True)
            conc = gal_batch.concentration()
            asym = gal_batch.asymmetry()
            m20 = gal_batch.m_20()
            batch_results = pd.DataFrame({'Gini': gini, 'C': conc, 'A': asym, 'M20': m20})
            results.append(batch_results)
            del images, gal_batch, gini, conc, asym, m20, batch_results

    df = pd.concat(results, ignore_index=True)
    return df

def image_mag(image, pixel_scale, z, zp, cosmo=cosmo):


    # check the units on pixel_scale
    if pixel_scale.unit == u.kpc:
        pixel_scale = pixel_scale*u.arcsec

        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)
        ckpc_per_arcsec = cosmo.kpc_comoving_per_arcmin(z).to(u.kpc/u.arcsec)

        ckpc_per_kpc = ckpc_per_arcsec / kpc_per_arcsec

        #define pixel scale of 0.05 kpc/pixel  
        pixel_scale = pixel_scale*u.kpc
        pixel_scale_comoving = pixel_scale * ckpc_per_kpc

        pixel_scale_arcsec = pixel_scale_comoving/ckpc_per_arcsec
    else:
        pixel_scale_arcsec = pixel_scale


    image = image - 2.5*np.log10(pixel_scale_arcsec.value**2)
    
    image_cps = data_util.magnitude2cps(image, zp)

    m = -2.5*np.log10(np.sum(10**(-0.4*image)))

    return m, image_cps

def preprocess(images, snapshot):
    snapshot_redshift = {
        0: 20.0464909888075,
        1: 14.9891732400424,
        2: 11.9802133153003,
        3: 10.9756432941379,
        4: 9.99659046618633,
        5: 9.38877127194055,
        6: 9.00233985416247,
        7: 8.44947629436874,
        8: 8.01217294886593,
        9: 7.5951071498716,
        10: 7.23627606616736,
        11: 7.00541704554453,
        12: 6.4915977456675,
        13: 6.0107573988449,
        14: 5.84661374788187,
        15: 5.5297658079491,
        16: 5.22758097312734,
        17: 4.99593346816462,
        18: 4.66451770247093,
        19: 4.42803373660555,
        20: 4.17683491472647,
        21: 4.00794511146527,
        22: 3.70877426464224,
        23: 3.49086136926065,
        24: 3.28303305795652,
        25: 3.00813107163038,
        26: 2.89578500572743,
        27: 2.73314261731872,
        28: 2.57729027160189,
        29: 2.44422570455415,
        30: 2.31611074395689,
        31: 2.2079254723837,
        32: 2.10326965259577,
        33: 2.00202813925285,
        34: 1.90408954353277,
        35: 1.82268925262035,
        36: 1.74357057433086,
        37: 1.66666955611447,
        38: 1.60423452207311,
        39: 1.53123902915761,
        40: 1.49551216649556,
        41: 1.41409822037252,
        42: 1.357576667403,
        43: 1.30237845990597,
        44: 1.24847261424514,
        45: 1.206258080781,
        46: 1.15460271236022,
        47: 1.11415056376538,
        48: 1.07445789454767,
        49: 1.03551044566414,
        50: 0.99729422578194,
        51: 0.950531351585033,
        52: 0.923000816177909,
        53: 0.886896937575248,
        54: 0.851470900624649,
        55: 0.816709979011851,
        56: 0.791068248946339,
        57: 0.757441372615853,
        58: 0.732636182022312,
        59: 0.700106353718523,
        60: 0.676110411213478,
        61: 0.644641840684537,
        62: 0.621428745242514,
        63: 0.598543288187567,
        64: 0.575980845107887,
        65: 0.546392183141022,
        66: 0.524565820433923,
        67: 0.503047523244883,
        68: 0.481832943420951,
        69: 0.460917794180647,
        70: 0.440297849247743,
        71: 0.419968941997267,
        72: 0.399926964613563,
        73: 0.380167867260239,
        74: 0.360687657261817,
        75: 0.347853841858178,
        76: 0.328829724205954,
        77: 0.310074120127834,
        78: 0.297717684517447,
        79: 0.27335334657844,
        80: 0.261343256161012,
        81: 0.24354018155467,
        82: 0.225988386260198,
        83: 0.214425035514495,
        84: 0.19728418237601,
        85: 0.180385261705749,
        86: 0.169252033243611,
        87: 0.152748768902381,
        88: 0.141876203969562,
        89: 0.125759332411261,
        90: 0.109869940458825,
        91: 0.0994018026302219,
        92: 0.0838844307974793,
        93: 0.0736613846564387,
        94: 0.058507322794513,
        95: 0.0485236299818059,
        96: 0.0337243718735154,
        97: 0.0239744283827625,
        98: 0.00952166696794476,
        99: 2.22044604925031e-16
    }
    print(f"Dataset shape: {images.shape}")

    pixel_scale = 100/images.shape[-1]*u.kpc

    # angular diameter distance
    d_A = cosmo.angular_diameter_distance(snapshot_redshift[33]).to(u.kpc)

    pixel_scale_arcsec = pixel_scale / d_A * u.rad.to(u.arcsec)

    print(f"pixel scale: {pixel_scale_arcsec}")

    mag, galaxy_image_cps_z2 = image_mag(images- 2.5*np.log10(pixel_scale_arcsec**2), pixel_scale_arcsec, snapshot_redshift[snapshot], 26.0)


    return mag, galaxy_image_cps_z2

def convolve_PSF(images, FWHM):

    snapshot_redshift = {
        0: 20.0464909888075,
        1: 14.9891732400424,
        2: 11.9802133153003,
        3: 10.9756432941379,
        4: 9.99659046618633,
        5: 9.38877127194055,
        6: 9.00233985416247,
        7: 8.44947629436874,
        8: 8.01217294886593,
        9: 7.5951071498716,
        10: 7.23627606616736,
        11: 7.00541704554453,
        12: 6.4915977456675,
        13: 6.0107573988449,
        14: 5.84661374788187,
        15: 5.5297658079491,
        16: 5.22758097312734,
        17: 4.99593346816462,
        18: 4.66451770247093,
        19: 4.42803373660555,
        20: 4.17683491472647,
        21: 4.00794511146527,
        22: 3.70877426464224,
        23: 3.49086136926065,
        24: 3.28303305795652,
        25: 3.00813107163038,
        26: 2.89578500572743,
        27: 2.73314261731872,
        28: 2.57729027160189,
        29: 2.44422570455415,
        30: 2.31611074395689,
        31: 2.2079254723837,
        32: 2.10326965259577,
        33: 2.00202813925285,
        34: 1.90408954353277,
        35: 1.82268925262035,
        36: 1.74357057433086,
        37: 1.66666955611447,
        38: 1.60423452207311,
        39: 1.53123902915761,
        40: 1.49551216649556,
        41: 1.41409822037252,
        42: 1.357576667403,
        43: 1.30237845990597,
        44: 1.24847261424514,
        45: 1.206258080781,
        46: 1.15460271236022,
        47: 1.11415056376538,
        48: 1.07445789454767,
        49: 1.03551044566414,
        50: 0.99729422578194,
        51: 0.950531351585033,
        52: 0.923000816177909,
        53: 0.886896937575248,
        54: 0.851470900624649,
        55: 0.816709979011851,
        56: 0.791068248946339,
        57: 0.757441372615853,
        58: 0.732636182022312,
        59: 0.700106353718523,
        60: 0.676110411213478,
        61: 0.644641840684537,
        62: 0.621428745242514,
        63: 0.598543288187567,
        64: 0.575980845107887,
        65: 0.546392183141022,
        66: 0.524565820433923,
        67: 0.503047523244883,
        68: 0.481832943420951,
        69: 0.460917794180647,
        70: 0.440297849247743,
        71: 0.419968941997267,
        72: 0.399926964613563,
        73: 0.380167867260239,
        74: 0.360687657261817,
        75: 0.347853841858178,
        76: 0.328829724205954,
        77: 0.310074120127834,
        78: 0.297717684517447,
        79: 0.27335334657844,
        80: 0.261343256161012,
        81: 0.24354018155467,
        82: 0.225988386260198,
        83: 0.214425035514495,
        84: 0.19728418237601,
        85: 0.180385261705749,
        86: 0.169252033243611,
        87: 0.152748768902381,
        88: 0.141876203969562,
        89: 0.125759332411261,
        90: 0.109869940458825,
        91: 0.0994018026302219,
        92: 0.0838844307974793,
        93: 0.0736613846564387,
        94: 0.058507322794513,
        95: 0.0485236299818059,
        96: 0.0337243718735154,
        97: 0.0239744283827625,
        98: 0.00952166696794476,
        99: 2.22044604925031e-16
    }
    
    pixel_scale = 100/images.shape[-1]*u.kpc

    # angular diameter distance
    d_A = cosmo.angular_diameter_distance(snapshot_redshift[33]).to(u.kpc)

    pixel_scale_arcsec = pixel_scale / d_A * u.rad.to(u.arcsec)

    # convolve the image with a gaussian kernel defined by the PSF

    FWHM_pix = FWHM/pixel_scale_arcsec
    sigma = (FWHM_pix/(2*np.sqrt(2*np.log(2)))).value

    images = scipy.ndimage.gaussian_filter(images, sigma=sigma)

    return images

if __name__ == "__main__":
    images = np.load('data/source_images_50_jwstf444w.npy')
    _ , images = preprocess(images, 50)

    FWHM = [0, 0.01, 0.08, 0.15]

    for i in FWHM:

        gal_batch = convolve_PSF(images, i)

        np.save(f"data/convolved_images_{i}.npy",gal_batch)

        gal_batch = Galaxy(gal_batch)

        np.save(f"data/segmentation_maps_{i}.npy",gal_batch.segmentation_map)
        gini = gal_batch.gini(segmap=True)
        conc = gal_batch.concentration()
        asym = gal_batch.asymmetry()
        m20 = gal_batch.m_20()
        df = batch_results = pd.DataFrame({'Gini': gini, 'C': conc, 'A': asym, 'M20': m20})
        df.to_csv(f'data/source_images_50_jwstf444w_morphology_{i}.csv', index=False)

        plot = sns.pairplot(df, corner=True)
        #Save the plot
        plt.savefig(f'data/source_images_50_jwstf444w_pairplot_{i}.png')