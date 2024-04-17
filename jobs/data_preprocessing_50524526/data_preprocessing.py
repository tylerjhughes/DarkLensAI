import pandas as pd
import numpy as np
import h5py

def preprocess(data):
    # Mean Normalization
    data = ((data-np.mean(data[:], axis=(2, 3))[:, :, None, None])/np.std(data[:], axis=(2, 3))[:, :, None, None])

    # Reshape data
    input_shape = (np.shape(data[0])[1], np.shape(data[0])[2], 1) 
    data = data.reshape(len(data), input_shape[0], input_shape[1], input_shape[2])
    #noise = noise.reshape(len(noise), input_shape[0], input_shape[1], input_shape[2])

    return data

with h5py.File('data/dataset.h5', 'a') as f:
    
    # Get the shape of the two datasets
    ds_perturbed_shape = f['CONFIGURATION_1_images'].shape
    ds_unperturbed_shape = f['CONFIGURATION_2_images'].shape

    # Create a new dataset for the preprocessed images if it doesn't exist
    if 'CONFIGURATION_1_images_preprocessed' in f:
        del f['CONFIGURATION_1_images_preprocessed']
    if 'CONFIGURATION_2_images_preprocessed' in f:
        del f['CONFIGURATION_2_images_preprocessed']

    f.create_dataset('CONFIGURATION_1_images_preprocessed', ds_perturbed_shape, dtype='f')
    f.create_dataset('CONFIGURATION_2_images_preprocessed', ds_unperturbed_shape, dtype='f')

    # Preprocess the images in batches
    batch_size = 1000
    for i in range(0, ds_perturbed_shape[0], batch_size):
        # if i % 10000 == 0:
        print(f'Preprocessing perturbed images: {i}/{ds_perturbed_shape[0]}')

        try:     
            perturbed_images = f['CONFIGURATION_1_images'][i:i+batch_size]
            perturbed_images = preprocess(perturbed_images)
            f['CONFIGURATION_1_images_preprocessed'][i:i+batch_size] = perturbed_images
        except:
            perturbed_images = f['CONFIGURATION_1_images'][i:]
            perturbed_images = preprocess(perturbed_images)
            f['CONFIGURATION_1_images_preprocessed'][i:] = perturbed_images

    for i in range(0, ds_unperturbed_shape[0], batch_size):
        if i % 10000 == 0:
            print(f'Preprocessing perturbed images: {i}/{ds_perturbed_shape[0]}')
        
        try:
            unperturbed_images = f['CONFIGURATION_2_images'][i:i+batch_size]
            unperturbed_images = preprocess(unperturbed_images)
            f['CONFIGURATION_2_images_preprocessed'][i:i+batch_size] = unperturbed_images
        except:
            unperturbed_images = f['CONFIGURATION_2_images'][i:]
            unperturbed_images = preprocess(unperturbed_images)
            f['CONFIGURATION_2_images_preprocessed'][i:] = unperturbed_images

