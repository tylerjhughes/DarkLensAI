print('Starting')
import h5py
import numpy as np

def preprocess(data):
    # Mean Normalization
    data = ((data-np.mean(data[:], axis=(2, 3))[:, :, None, None])/np.std(data[:], axis=(2, 3))[:, :, None, None])

    # Reshape data
    input_shape = (np.shape(data[0])[1], np.shape(data[0])[2], 1) 
    data = data.reshape(len(data), input_shape[0], input_shape[1], input_shape[2])
    #noise = noise.reshape(len(noise), input_shape[0], input_shape[1], input_shape[2])

    return data

def split_hdf5_file(input_file_path, output_file_prefix, num_splits, batch_size=10000):
    with h5py.File(input_file_path, 'r') as input_file:
        config_1_images = input_file['CONFIGURATION_1_images']
        config_2_images = input_file['CONFIGURATION_2_images']

        total_samples = len(config_1_images)
        samples_per_split = 250000

        for i in range(num_splits):
            print(f"Splitting part {i+1}/{num_splits}", end='               \r')
            start_idx = i * samples_per_split
            end_idx = (i + 1) * samples_per_split if i != num_splits - 1 else total_samples

            output_file_path = f"{output_file_prefix}dataset_sie_part_{i+1}.h5"
            with h5py.File(output_file_path, 'w') as output_file:
                
                # Get the shape of the images
                image_shape = config_1_images.shape[1:]

                # Reorder the shape to have the channel as the last dimension
                dset_shape = (end_idx - start_idx,) + image_shape[1:] + (image_shape[0],) 

                config_1_dset = output_file.create_dataset('CONFIGURATION_1_images_preprocessed', dset_shape, dtype=config_1_images.dtype)
                config_2_dset = output_file.create_dataset('CONFIGURATION_2_images_preprocessed', dset_shape, dtype=config_2_images.dtype)

                for batch_start in range(start_idx, end_idx, batch_size):
                    batch_end = min(batch_start + batch_size, end_idx)
                    #print(f"Processing batch {batch_start} to {batch_end}", end='               \r')

                    batch_config_1_preprocessed = preprocess(config_1_images[batch_start:batch_end])
                    batch_config_2_preprocessed = preprocess(config_2_images[batch_start:batch_end])

                    config_1_dset[batch_start - start_idx:batch_end - start_idx] = batch_config_1_preprocessed
                    config_2_dset[batch_start - start_idx:batch_end - start_idx] = batch_config_2_preprocessed
                    
                    print(f"Completed batch {batch_start} to {batch_end}", end='               \r')
                
                # close the file
                output_file.close()

    print(f"Dataset split into {num_splits} parts.")

# Example usage
input_file_path = 'data/sie/dataset_sie.h5'
output_file_prefix = 'data/sie/'
num_splits = 17  # Number of parts to split the dataset into

split_hdf5_file(input_file_path, output_file_prefix, num_splits)