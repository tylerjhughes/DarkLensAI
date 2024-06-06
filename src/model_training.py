from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, Callback
import numpy as np
import os
import sys
import h5py
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parallel_hdf5_generator(file_paths, batch_size, start_idx, end_idx, limit, is_test=False):
    current_file = 0
    current_start_idx = start_idx
    processed_samples = 0
    while True:
        if processed_samples >= limit:
            break
        h5_file_path = file_paths[current_file]
        with h5py.File(h5_file_path, 'r') as f:
            total_samples = 500000
            batch_end_idx = min(current_start_idx + batch_size, total_samples, end_idx)

            if current_start_idx < batch_end_idx:

                perturbed_images = f['CONFIGURATION_1_images_preprocessed'][current_start_idx//2:batch_end_idx//2]
                unperturbed_images = f['CONFIGURATION_2_images_preprocessed'][current_start_idx//2:batch_end_idx//2]

                assert len(unperturbed_images) == len(perturbed_images), "Mismatch in dataset lengths"

                dataset = np.concatenate((perturbed_images, unperturbed_images), axis=0)
                labels = np.concatenate((np.ones(len(perturbed_images)), np.zeros(len(unperturbed_images))))

                if is_test:
                    print(f"Current file: {current_file}, Processed Samples: {processed_samples}, Current start idx: {current_start_idx}, Batch end idx: {batch_end_idx}")
                    # Check for nan values
                    if np.isnan(labels).any():
                        print("Nan values found in labels")
                        sys.exit()
                    if np.isnan(dataset).any():
                        print("Nan values found in dataset")
                        sys.exit()

                # shuffle
                indices = np.arange(len(dataset))
                np.random.shuffle(indices)
                dataset = dataset[indices]
                labels = labels[indices]

                assert len(dataset) == len(labels), "Mismatch in dataset and label lengths"

                yield dataset, labels
                current_start_idx = batch_end_idx
                processed_samples += len(dataset)
            else:
                if is_test:
                    print(f"IF FAILED: Current file: {current_file}, Processed Samples: {processed_samples}, Current start idx: {current_start_idx}, Batch end idx: {batch_end_idx}")
                current_file += 1
                current_start_idx = 0 
                f.close() 
            # close the file
            f.close()
    if is_test:
        print(f"Processed {processed_samples} samples")
    f.close()
    current_file = 0
    current_start_idx = start_idx


STARTTIME = datetime.now()
DATASET_PATH = str(sys.argv[1])
LR = float(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
OUT_DIR = str(sys.argv[4])
MODEL = str(sys.argv[5])
DATASET_SIZE = int(sys.argv[6])
NUM_SPLITS = 17

# Split the dataset size into training (80%) and validation (20%) sizes
train_size = int(0.8 * DATASET_SIZE)
val_size = DATASET_SIZE - train_size

# Create file paths for the splits
file_paths = [f"{DATASET_PATH}dataset_sie_part_{i+1}.h5" for i in range(NUM_SPLITS)]

# Calculate the start and end indices for training and validation data
train_end_file_idx = train_size // 500000
train_end_idx_in_last_file = train_size % 500000

val_start_file_idx = train_end_file_idx
val_start_idx_in_first_val_file = train_end_idx_in_last_file

print(f"Train end file idx: {train_end_file_idx}")
print(f"Train end idx in last file: {train_end_idx_in_last_file}")

print(f"Training data: {train_size} samples, Validation data: {val_size} samples")
print(f"Training data: {len(file_paths)} files, Validation data: {len(file_paths[val_start_file_idx:])} files")

# Get image shape
with h5py.File(file_paths[0], 'r') as f:
    try:
        image_shape = f['CONFIGURATION_1_images_preprocessed'].shape[1:]
    except KeyError:
        print("KeyError: 'CONFIGURATION_1_images_preprocessed' not found in dataset. Ensure you are using the correct dataset.")
        sys.exit()

# Initialize the ResNet50 model
model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=image_shape,
    pooling=None,
    classes=2,
    classifier_activation='softmax'
)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

# Make it a dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: parallel_hdf5_generator(file_paths[:train_end_file_idx+1], BATCH_SIZE, 0, train_size, train_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=([None, image_shape[0], image_shape[1], image_shape[2]], [None])
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: parallel_hdf5_generator(file_paths[val_start_file_idx:], BATCH_SIZE, val_start_idx_in_first_val_file, train_size + val_size, val_size),
    output_types=(tf.float32, tf.int32),
    output_shapes=([None, image_shape[0], image_shape[1], image_shape[2]], [None])
).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define callbacks
csv_logger = CSVLogger(f'{OUT_DIR}/trainingLog.log', append=True)

model_checkpoint_callback = ModelCheckpoint(
    filepath=OUT_DIR + "/checkpoint_{epoch:02d}.keras",
    save_weights_only=False,
    monitor='val_accuracy',
    save_freq='epoch'
)

# Setup TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=f"{OUT_DIR}/logs",  # Specify the path to save the logs
    histogram_freq=1,  # Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed.
    write_graph=True,  # Enables visualization of the graph in TensorBoard
    update_freq='epoch'  # Updates the logs after each epoch
)

model.fit(
    train_dataset,
    epochs=500,
    validation_data=validation_dataset,
    callbacks=[csv_logger, model_checkpoint_callback, tensorboard_callback],
    verbose=2
)

ENDTIME = datetime.now()
print(f"Training complete. Time taken: {ENDTIME - STARTTIME}")
print("Script Complete")
