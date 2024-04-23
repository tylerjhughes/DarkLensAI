import sys
import os
from utilities import YamlEditor
import deeplenstronomy.deeplenstronomy as dl
import pandas as pd
import numpy as np
import time
import h5py
import fcntl

## FUNCTIONS

def append_data_to_file(dataset, dataset_path):
    WAIT_TIME = 5
    lock_file = "jobs/running/file.lock"

    while True:
        try:
            # Try to create the lock file
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # If successful, close the file descriptor and proceed
            os.close(fd)

            # Open the HDF5 file in append mode
            with h5py.File(dataset_path, 'a') as f:
                # Append your data to the HDF5 file
                for item in [x for x in dir(dataset) if x[0:13] == 'CONFIGURATION']:
                    if item != 'CONFIGURATION_2_unperturbed_images':
                        try:
                            data = getattr(dataset, item)
                        except KeyError:
                            print(f"Attribute '{item}' not found in dataset, skipping.")
                            continue     

                        if isinstance(data, pd.DataFrame):
                            # Add LENS_ID_ARRAY as the index
                            data['LENS_ID'] = LENS_ID_ARRAY
                            data.set_index('LENS_ID', inplace=True)
                            print(f"Saving {item} to jobs/{JOB_NAME}/{JOB_ARRAY_ID}/{item}.csv")
                            # Check if file exists
                            if os.path.exists(f'data/{item}.csv'):
                                # Append data to existing file
                                data.to_csv(f'data/{item}.csv', mode='a', header=False)
                            else:
                                data.to_csv(f'data/{item}.csv')
                        else: 
                            if item in f:
                                # append item to existing dataset
                                f[item].resize((f[item].shape[0] + data.shape[0]), axis=0)
                                f[item][-data.shape[0]:] = data.astype(np.float32)
                            else:
                                f.create_dataset(item, data=data.astype(np.float32), maxshape=(None, *data.shape[1:]), chunks=True) 

                if 'LENS_ID' not in f:
                    f.create_dataset('LENS_ID', data=LENS_ID_ARRAY, chunks=True, maxshape=(None,))
                else:
                    try:
                        f['LENS_ID'].resize(f['LENS_ID'].shape[0] + LENS_ID_ARRAY.shape[0], axis=0)
                    except KeyError:
                        # 'LENS_ID' doesn't exist, create it
                        f.create_dataset('LENS_ID', data=LENS_ID_ARRAY)
                    else:
                        f['LENS_ID'][-LENS_ID_ARRAY.shape[0]:] = LENS_ID_ARRAY

                f.close()
        except FileExistsError:
            # If the lock file already exists, wait and try again
            print(f"File is locked, waiting {WAIT_TIME} seconds...")
            time.sleep(WAIT_TIME)
            continue
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # This block executes whether an exception is thrown or not
            if os.path.exists(lock_file):
                os.remove(lock_file)
            break  # Exit loop after cleaning up

## VARIABLES
DATASET_SIZE = int(sys.argv[1])
RESOLUTION = float(sys.argv[2])
JOB_NAME = sys.argv[3]
JOB_ARRAY_ID = int(sys.argv[4])

print(DATASET_SIZE)

if JOB_ARRAY_ID != None:
    CONFIG_FILE_PATH = f'jobs/{JOB_NAME}/{JOB_ARRAY_ID}/config_files/config.yaml'
else:
    CONFIG_FILE_PATH = f'jobs/{JOB_NAME}/config_files/config.yaml'

DATASET_PATH = 'data/dataset.h5'

LENS_ID_ARRAY = np.arange(int(DATASET_SIZE/2)) + JOB_ARRAY_ID*int(DATASET_SIZE/2)

## UPDATE CONFIG FILES
# print the current working directory
print(os.getcwd())

yamleditor = YamlEditor(CONFIG_FILE_PATH)

yamleditor.update_yaml({'DATASET': {'PARAMETERS': {'SIZE': DATASET_SIZE+2}}})
yamleditor.update_yaml({'DATASET': {'PARAMETERS': {'OUTDIR': f"jobs/{JOB_NAME}/{JOB_ARRAY_ID}"}}})
yamleditor.update_yaml({'DATASET': {'PARAMETERS': {'SEED': 700 + JOB_ARRAY_ID}}})
yamleditor.update_yaml({'SURVEY': {'PARAMETERS': {'seeing': RESOLUTION}}})


## GENERATE DATA
DATASET = dl.make_dataset(CONFIG_FILE_PATH, verbose=True)

print(f"SEED: {JOB_ARRAY_ID}")
## SAVE DATA TO HDF5 FILE
append_data_to_file(DATASET, DATASET_PATH)