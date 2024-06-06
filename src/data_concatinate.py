import h5py
import numpy as np

# Define the size of the chunks
chunk_size = 1000

# Open both files
with h5py.File('data/dataset_epl.h5', 'a') as epl, h5py.File('data/dataset.h5', 'r') as ds:
    # For each key, append the data from dataset.h5 to dataset_epl.h5
    for key in ['CONFIGURATION_1_images', 'CONFIGURATION_2_images']:

        print(f"Processing {key}...")

        # Get the shape of the data in dataset.h5
        shape_ds = ds[key].shape
        
        # Resize the dataset in dataset_epl.h5 to accommodate the new data
        shape_epl = epl[key].shape
        epl[key].resize((shape_epl[0] + shape_ds[0],) + shape_epl[1:])
        
        # Append the data from dataset.h5 to dataset_epl.h5 in chunks
        for i in range(0, shape_ds[0], chunk_size):
            print(f"Percent complete: {i/shape_ds[0]*100:.2f}%", end="\r")
            end_index = min(i + chunk_size, shape_ds[0])
            chunk = ds[key][i:end_index]
            epl[key][shape_epl[0]+i:shape_epl[0]+end_index] = chunk