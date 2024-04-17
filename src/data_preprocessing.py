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

with h5py.File('data/dataset.h5', 'r') as f:
    ds_perturbed_shape = f['CONFIGURATION_1_images'].shape
    ds_unperturbed_shape = f['CONFIGURATION_2_images'].shape

    with h5py.File('data/dataset_preprocessed.h5', 'a') as hf:
        if 'CONFIGURATION_1_images_preprocessed' not in hf:
            hf.create_dataset('CONFIGURATION_1_images_preprocessed', ds_perturbed_shape, dtype='f')
        if 'CONFIGURATION_2_images_preprocessed' not in hf:
            hf.create_dataset('CONFIGURATION_2_images_preprocessed', ds_unperturbed_shape, dtype='f')

        batch_size = 1000
        for i in range(0, ds_perturbed_shape[0], batch_size):
            print(f'Preprocessing perturbed images: {i}/{ds_perturbed_shape[0]}')
            end_index = min(i + batch_size, ds_perturbed_shape[0])
            perturbed_images = f['CONFIGURATION_1_images'][i:end_index]
            perturbed_images = preprocess(perturbed_images)
            hf['CONFIGURATION_1_images_preprocessed'][i:end_index] = perturbed_images

        for i in range(0, ds_unperturbed_shape[0], batch_size):
            print(f'Preprocessing unperturbed images: {i}/{ds_unperturbed_shape[0]}')
            end_index = min(i + batch_size, ds_unperturbed_shape[0])
            unperturbed_images = f['CONFIGURATION_2_images'][i:end_index]
            unperturbed_images = preprocess(unperturbed_images)
            hf['CONFIGURATION_2_images_preprocessed'][i:end_index] = unperturbed_images

